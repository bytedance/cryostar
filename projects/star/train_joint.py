import os.path as osp
from pathlib import Path
import warnings
from copy import deepcopy
import collections

import einops
import numpy as np
import biotite.structure as struc

import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torchinfo import summary

import lightning.pytorch as pl
from lightning.fabric.utilities.warnings import PossibleUserWarning
from lightning.pytorch.utilities import rank_zero_only
from lightning.pytorch.strategies import DDPStrategy

from mmengine import mkdir_or_exist

from cryostar.utils.transforms import SpatialGridTranslate
# other
from cryostar.utils.dataio import StarfileDataSet, StarfileDatasetConfig, Mask
from cryostar.utils.ctf_utils import CTFRelion, CTFCryoDRGN
from cryostar.utils.losses import calc_cor_loss, calc_kl_loss
from cryostar.utils.misc import log_to_current, \
    pl_init_exp, pretty_dict, set_seed, warmup, create_circular_mask
from cryostar.utils.pdb_tools import bt_save_pdb
from cryostar.gmm.gmm import EMAN2Grid, batch_projection, Gaussian
from cryostar.gmm.deformer import E3Deformer, NMADeformer
from cryostar.utils.fft_utils import primal_to_fourier_2d, fourier_to_primal_2d
from cryostar.utils.polymer import Polymer, NT_ATOMS, AA_ATOMS
from cryostar.utils.dist_loss import (find_quaint_cutoff_pairs, find_range_cutoff_pairs, find_continuous_pairs,
                                      calc_dist_by_pair_indices, remove_duplicate_pairs, filter_same_chain_pairs,
                                      DistLoss)
from cryostar.utils.latent_space_utils import get_nearest_point, cluster_kmeans, run_pca, get_pc_traj, run_umap
from cryostar.utils.vis_utils import plot_z_dist, save_tensor_image
from cryostar.utils.pl_utils import merge_step_outputs, squeeze_dict_outputs_1st_dim, \
    filter_outputs_by_indices, get_1st_unique_indices
from cryostar.utils.mrc_tools import save_mrc

from miscs import calc_pair_dist_loss, calc_clash_loss, low_pass_mask2d, infer_ctf_params_from_config, GMMVolVAE

# avoid num_workers set as cpu_count warning
warnings.simplefilter("ignore", PossibleUserWarning)

# only log to rank_zero, comment this for debugging
log_to_current = rank_zero_only(log_to_current)

TASK_NAME = "joint"


def prepare_images(images: torch.FloatTensor, space: str):
    assert space in ("real", "fourier")
    if space == "real":
        model_input = einops.rearrange(images, "b 1 ny nx -> b (1 ny nx)")
    else:
        fimages = primal_to_fourier_2d(images)
        model_input = einops.rearrange(torch.view_as_real(fimages), "b 1 ny nx c2 -> b (1 ny nx c2)", c2=2)
    return model_input


class InitTask(pl.LightningModule):

    def __init__(self, em_module):
        super().__init__()
        self.cfg = em_module.cfg
        self.em_module = em_module
        self.loss_deque = collections.deque([
            10,
        ], maxlen=20)

    def on_train_batch_end(self, outputs, batch, batch_idx):
        self.loss_deque.append(outputs['loss'].item())
        if np.mean(self.loss_deque) < 1e-3:
            self.trainer.should_stop = True
        # update all process status
        self.trainer.should_stop = self.trainer.strategy.broadcast(self.trainer.should_stop)

    def training_step(self, batch, batch_idx):
        images = batch["proj"]
        idxes = batch["idx"]
        rot_mats, trans_mats = self.em_module.get_batch_pose(batch)

        pred_deformation, pred_f_slice, mu, log_var = self.em_module.model(images.flatten(start_dim=1), idxes, rot_mats)

        shift_loss = torch.mean(torch.pow(pred_deformation.flatten(start_dim=-2), 2))
        loss = shift_loss
        if self.global_step % self.cfg.runner.log_every_n_step == 0:
            log_to_current(f"loss {loss.item()}")
        return loss

    def configure_optimizers(self):
        return optim.AdamW(self.em_module.model.parameters(), lr=1e-4)

    def on_fit_end(self):
        log_to_current(f"Init finished with loss {np.mean(self.loss_deque)}")


class CryoEMTask(pl.LightningModule):

    def __init__(self, cfg, dataset):
        super().__init__()
        cfg = deepcopy(cfg)
        self.cfg = cfg

        # Define GMM
        meta = Polymer.from_pdb(cfg.dataset_attr.ref_pdb_path)
        log_to_current(f"Load reference structure from {cfg.dataset_attr.ref_pdb_path}")

        # for save
        self.template_pdb = meta.to_atom_arr()
        log_to_current(f"Protein contains {len(meta)} atoms, "
                       f"{meta.num_amino_acids} amino acids, "
                       f"{meta.num_nucleotides} nucleotides, "
                       f"{meta.num_chains} chains.")

        # ref
        ref_centers = torch.from_numpy(meta.coord).float()
        ref_amps = torch.from_numpy(meta.num_electron).float()
        ref_sigmas = torch.ones_like(ref_amps)
        ref_sigmas.fill_(2.)
        log_to_current(f"1st GMM blob amplitude {ref_amps[0].item()}, sigma {ref_sigmas[0].item()}")

        num_pts = len(meta)
        log_to_current(f"Reference structure has {num_pts} atom coordinates")

        # tunable params
        # gmm
        self.register_buffer("gmm_centers", ref_centers)

        if cfg.gmm.tunable:
            log_to_current("Set GMM sigmas, amplitudes tunable")
            self.register_parameter("gmm_sigmas", nn.Parameter(ref_sigmas))
            self.register_parameter("gmm_amps", nn.Parameter(ref_amps))
        else:
            self.register_buffer("gmm_sigmas", ref_sigmas)
            self.register_buffer("gmm_amps", ref_amps)

        nma_modes = None
        if (hasattr(self.cfg.extra_input_data_attr, "nma_path") and
                self.cfg.extra_input_data_attr.nma_path not in ["", None]):
            nma_modes = torch.tensor(np.load(self.cfg.extra_input_data_attr.nma_path), dtype=torch.float32)
            log_to_current(f"Load NMA coefficients from {self.cfg.extra_input_data_attr.nma_path}, "
                           f"whose shape is {nma_modes.shape}")

        # model
        if cfg.model.input_space == "fourier":
            in_dim = 2 * cfg.data_process.down_side_shape ** 2
        elif cfg.model.input_space == "real":
            in_dim = cfg.data_process.down_side_shape ** 2
        else:
            raise NotImplementedError

        cfg.model.model_cfg.vol_params.update(
            {
                "pe_dim": self.cfg.data_process.down_side_shape,
                "D": self.cfg.data_process.down_side_shape,
                "force_symmetry": False,
            }
        )

        self.model = GMMVolVAE(in_dim=in_dim, out_dim=num_pts * 3 if nma_modes is None else 6 + nma_modes.shape[1],
                               side_shape=cfg.data_process.down_side_shape,
                               **cfg.model.model_cfg)
        # self.model = VAE(in_dim=in_dim,
        #                  out_dim=num_pts * 3 if nma_modes is None else 6 + nma_modes.shape[1],
        #                  **cfg.model.model_cfg)
        # log_to_current('Model summary:\n' +
        #                str(summary(self.model, input_size=[(1, cfg.data_process.down_side_shape ** 2), (1,), (1, 3, 3)],
        #                            verbose=0)))
        if nma_modes is None:
            self.deformer = E3Deformer()
        else:
            self.deformer = NMADeformer(nma_modes)

        # loss or regularization's preparation
        # dist loss
        connect_pairs = find_continuous_pairs(meta.chain_id, meta.res_id, meta.atom_name)

        if cfg.extra_input_data_attr.use_domain:
            log_to_current("use domain instead of chain!")
            domain_id = np.load(cfg.extra_input_data_attr.domain_path)
            cutoff_pairs = find_quaint_cutoff_pairs(meta.coord, domain_id, meta.res_id, cfg.loss.intra_chain_cutoff,
                                                    cfg.loss.inter_chain_cutoff, cfg.loss.intra_chain_res_bound)
        else:
            # deal with RNA/DNA
            if np.sum(np.isin(meta.atom_name, NT_ATOMS)):
                # aa
                tmp_mask = np.isin(meta.atom_name, AA_ATOMS)
                indices_in_pdb = np.nonzero(tmp_mask)[0]
                aa_cutoff_pairs = find_quaint_cutoff_pairs(meta.coord[tmp_mask], meta.chain_id[tmp_mask],
                                                           meta.res_id[tmp_mask], cfg.loss.intra_chain_cutoff,
                                                           cfg.loss.inter_chain_cutoff, cfg.loss.intra_chain_res_bound)
                aa_cutoff_pairs = indices_in_pdb[aa_cutoff_pairs]
                log_to_current(f"{len(aa_cutoff_pairs)} AA pairs")
                # nt
                tmp_mask = np.isin(meta.atom_name, NT_ATOMS)
                indices_in_pdb = np.nonzero(tmp_mask)[0]
                nt_cutoff_pairs = find_quaint_cutoff_pairs(meta.coord[tmp_mask], meta.chain_id[tmp_mask],
                                                           meta.res_id[tmp_mask], cfg.loss.nt_intra_chain_cutoff,
                                                           cfg.loss.nt_inter_chain_cutoff,
                                                           cfg.loss.nt_intra_chain_res_bound)
                nt_cutoff_pairs = indices_in_pdb[nt_cutoff_pairs]
                log_to_current(f"{len(nt_cutoff_pairs)} NT pairs")
                cutoff_pairs = np.vstack((aa_cutoff_pairs, nt_cutoff_pairs))

            else:
                cutoff_pairs = find_quaint_cutoff_pairs(meta.coord, meta.chain_id, meta.res_id,
                                                        cfg.loss.intra_chain_cutoff, cfg.loss.inter_chain_cutoff,
                                                        cfg.loss.intra_chain_res_bound)
        cutoff_pairs = remove_duplicate_pairs(cutoff_pairs, connect_pairs)

        if cfg.loss.sse_weight != 0.0:
            log_to_current("use pseduo `sse` by building spatial/sequential edges")
            sse_pairs = find_quaint_cutoff_pairs(meta.coord, meta.chain_id, meta.res_id, cfg.loss.intra_chain_cutoff, 0,
                                                 20)
            cutoff_pairs = remove_duplicate_pairs(cutoff_pairs, sse_pairs)

        clash_pairs = find_range_cutoff_pairs(meta.coord, cfg.loss.clash_min_cutoff)
        clash_pairs = remove_duplicate_pairs(clash_pairs, connect_pairs)

        if len(connect_pairs) > 0:
            self.register_buffer("connect_pairs", torch.from_numpy(connect_pairs).long())
            dists = calc_dist_by_pair_indices(meta.coord, connect_pairs)
            self.register_buffer("connect_dists", torch.from_numpy(dists).float())
            log_to_current(f"found {len(connect_pairs)} connect_pairs")
        else:
            log_to_current("connect_pairs is empty")

        if cfg.loss.sse_weight != 0.0:
            self.register_buffer("sse_pairs", torch.from_numpy(sse_pairs).long())
            dists = calc_dist_by_pair_indices(meta.coord, sse_pairs)
            self.register_buffer("sse_dists", torch.from_numpy(dists).float())
            log_to_current(f"found {len(sse_pairs)} sse_pairs")

        if len(cutoff_pairs) > 0:
            dists = calc_dist_by_pair_indices(meta.coord, cutoff_pairs)
            log_to_current(f"found {len(cutoff_pairs)} cutoff_pairs")
            self.dist_loss_fn = DistLoss(cutoff_pairs, dists, reduction=None)

            # for chain-wise dropout
            cutoff_chain_mask = filter_same_chain_pairs(cutoff_pairs, meta.chain_id)
            self.register_buffer("cutoff_chain_mask", torch.from_numpy(cutoff_chain_mask))
        else:
            log_to_current("cutoff_pairs is empty")

        if len(clash_pairs) > 0:
            self.register_buffer("clash_pairs", torch.from_numpy(clash_pairs).long())
            log_to_current(f"found {len(clash_pairs)} clash_pairs")
        else:
            log_to_current("clash_pairs is empty")

        # low-pass filtering
        if hasattr(cfg.data_process, "low_pass_bandwidth"):
            log_to_current(f"Use low-pass filtering w/ {cfg.data_process.low_pass_bandwidth} A")
            lp_mask2d = low_pass_mask2d(cfg.data_process.down_side_shape, cfg.data_process.down_apix,
                                        cfg.data_process.low_pass_bandwidth)
            self.register_buffer("lp_mask2d", torch.from_numpy(lp_mask2d).float())
        else:
            self.lp_mask2d = None

        #
        self.mask = Mask(cfg.data_process.down_side_shape, rad=cfg.loss.mask_rad_for_image_loss)
        mask = create_circular_mask(self.cfg.data_process.down_side_shape, self.cfg.data_process.down_side_shape, None,
                                    self.cfg.data_process.down_side_shape // 2 * self.cfg.loss.mask_rad_for_image_loss, )
        self.register_buffer("image_mask", torch.from_numpy(mask))

        # for projection
        grid = EMAN2Grid(side_shape=cfg.data_process.down_side_shape, voxel_size=cfg.data_process.down_apix)
        self.grid = grid

        ctf_params = infer_ctf_params_from_config(cfg)
        if cfg.model.ctf == "v1":
            self.ctf = CTFRelion(**ctf_params, num_particles=len(dataset))
            log_to_current("We will deprecate `model.ctf=v1` in a future version, use `model.ctf=v2` instead.")
        elif cfg.model.ctf == "v2":
            self.ctf = CTFCryoDRGN(**ctf_params, num_particles=len(dataset))
        else:
            raise NotImplementedError
        log_to_current(ctf_params)

        # translate image helper
        self.translator = SpatialGridTranslate(D=cfg.data_process.down_side_shape, device=self.device)

        self.apix = self.cfg.data_process.down_apix
        # cache
        self.validation_step_outputs = []
        self.stored_metrics = {}
        self.history_saved_dirs = []

        if getattr(self.cfg.extra_input_data_attr, "ckpt_path", None) is not None:
            log_to_current(f"load checkpoint from {self.cfg.extra_input_data_attr.ckpt_path}")
            self._load_ckpt(self.cfg.extra_input_data_attr.ckpt_path)

    def _save_ckpt(self, ckpt_path):
        torch.save(
            {
                "model": self.model.state_dict(),
                "gmm_sigmas": self.gmm_sigmas.data,
                "gmm_amps": self.gmm_amps.data
            }, ckpt_path)

    def _load_ckpt(self, ckpt_path):
        state_dict = torch.load(ckpt_path, map_location=self.device)

        self.model.load_state_dict(state_dict["model"])
        if self.cfg.gmm.tunable:
            self.gmm_sigmas.data = state_dict["gmm_sigmas"]
            self.gmm_amps.data = state_dict["gmm_amps"]

    def _get_save_dir(self):
        save_dir = osp.join(self.cfg.work_dir, f"{self.current_epoch:04d}_{self.global_step:07d}")
        mkdir_or_exist(save_dir)
        return save_dir

    def low_pass_images(self, images):
        f_images = primal_to_fourier_2d(images)
        f_images = f_images * self.lp_mask2d
        images = fourier_to_primal_2d(f_images).real
        return images

    def get_batch_pose(self, batch):
        rot_mats = batch["rotmat"]
        # yx order
        trans_mats = torch.concat((batch["shiftY"].unsqueeze(1), batch["shiftX"].unsqueeze(1)), dim=1)
        trans_mats /= self.apix
        return rot_mats, trans_mats

    def _shared_forward(self, images, idxes, rots):
        # predict structure
        pred_deformation, pred_f_slice, mu, log_var = self.model(prepare_images(images, self.cfg.model.input_space), idxes, rots)
        return pred_deformation, pred_f_slice, mu, log_var

    def _shared_projection(self, pred_struc, rot_mats):
        pred_images = batch_projection(
            gauss=Gaussian(
                mus=pred_struc,
                sigmas=self.gmm_sigmas.unsqueeze(0),  # (b, num_centers)
                amplitudes=self.gmm_amps.unsqueeze(0)),
            rot_mats=rot_mats,
            line_grid=self.grid.line())
        pred_images = einops.rearrange(pred_images, 'b y x -> b 1 y x')
        return pred_images

    def _apply_ctf(self, batch, real_proj, freq_mask=None):
        f_proj = primal_to_fourier_2d(real_proj)
        f_proj = self._apply_ctf_f(batch, f_proj, freq_mask)
        # Note: here only use the real part
        proj = fourier_to_primal_2d(f_proj).real
        return proj

    def _apply_ctf_f(self, batch, f_proj, freq_mask=None):
        pred_ctf_params = {k: batch[k] for k in ('defocusU', 'defocusV', 'angleAstigmatism') if k in batch}
        f_proj = self.ctf(f_proj, batch['idx'], ctf_params=pred_ctf_params, mode="gt", frequency_marcher=None)
        if freq_mask is not None:
            f_proj = f_proj * self.lp_mask2d
        return f_proj

    def _shared_infer(self, batch):
        gt_images = batch["proj"]
        idxes = batch["idx"]

        rot_mats, trans_mats = self.get_batch_pose(batch)

        # if self.lp_mask2d is not None:
        #     gt_images = self.low_pass_images(gt_images)

        # prediction
        pred_deformation, pred_vol_f_images, mu, log_var = self._shared_forward(gt_images, idxes, rot_mats)
        pred_struc = self.deformer.transform(pred_deformation, self.gmm_centers)

        # get gmm projections
        pred_gmm_images = self._shared_projection(pred_struc, rot_mats)

        # apply ctf, low-pass
        pred_gmm_images = self._apply_ctf(batch, pred_gmm_images, self.lp_mask2d)
        pred_vol_f_images = self._apply_ctf_f(batch, pred_vol_f_images, None)

        if trans_mats is not None:
            gt_images = self.translator.transform(einops.rearrange(gt_images, "B 1 NY NX -> B NY NX"),
                                                  einops.rearrange(trans_mats, "B C2 -> B 1 C2"))

        return gt_images, pred_gmm_images, pred_vol_f_images, pred_struc, mu, log_var

    def _shared_decoding(self, z):
        with torch.no_grad():
            z = z.float().to(self.device)

            pred_deformation = self.model.decoder(z)
            pred_struc = self.deformer.transform(pred_deformation, self.gmm_centers)
            pred_struc = pred_struc.squeeze(0)

            # now make_volume doesn't support batch
            pred_vol = []
            b = z.shape[0]
            for i in range(b):
                pred_vol.append(self.model.vol.make_volume(z[i:i + 1]))

            # pred_vol = self.model.vol.make_volume(z)
        return pred_struc, pred_vol

    def _save_batched_strucs(self, pred_strucs, save_path):
        ref_atom_arr = self.template_pdb.copy()
        atom_arrs = []
        b = pred_strucs.shape[0]
        for i in range(b):
            tmp_struc = pred_strucs[i].cpu().numpy()
            tmp_atom_arr = ref_atom_arr.copy()
            tmp_atom_arr.coord = tmp_struc
            atom_arrs.append(tmp_atom_arr)

        bt_save_pdb(save_path, struc.stack(atom_arrs))

    def _save_batched_vols(self, pred_vols, save_dir):
        b = len(pred_vols)
        for i in range(b):
            density = pred_vols[i]
            voxel_size = self.cfg.data_process.down_apix
            origin = -self.cfg.data_process.down_side_shape // 2 * voxel_size

            density = density.cpu().numpy()
            save_mrc(density, osp.join(save_dir, f"vol_{i}.mrc"), voxel_size, origin)

    def _shared_image_check(self, total=25):
        mode = self.model.training
        # use validation or test set which not shuffled
        tmp_loader = self.trainer.val_dataloaders or self.trainer.test_dataloaders

        num = 0
        gt_images_list = []
        pred_gmm_images_list = []
        pred_vol_images_list = []
        self.model.eval()
        with torch.no_grad():
            for batch in tmp_loader:
                batch = self.trainer.strategy.batch_to_device(batch)

                gt_images, pred_gmm_images, pred_vol_f_images, _, mu, log_var = self._shared_infer(batch)

                pred_vol_images = fourier_to_primal_2d(pred_vol_f_images).real

                gt_images_list.append(gt_images)
                pred_gmm_images_list.append(pred_gmm_images)
                pred_vol_images_list.append(pred_vol_images)

                num += gt_images.shape[0]
                if num >= total:
                    break

        self.model.train(mode=mode)

        gt_images_list = torch.cat(gt_images_list, dim=0)[:total]
        pred_gmm_images_list = torch.cat(pred_gmm_images_list, dim=0)[:total]
        pred_vol_images_list = torch.cat(pred_vol_images_list, dim=0)[:total]

        save_dir = self._get_save_dir()

        save_tensor_image(gt_images_list, osp.join(save_dir, "input_image.png"))
        save_tensor_image(pred_gmm_images_list, osp.join(save_dir, "pred_gmm_image.png"), self.mask.mask)
        save_tensor_image(pred_vol_images_list, osp.join(save_dir, "pred_vol_image.png"), self.mask.mask)

    # standard hooks:
    def training_step(self, batch, batch_idx):
        cfg = self.cfg
        bsz = len(batch["proj"])

        gt_images, pred_gmm_images, pred_vol_f_images, pred_struc, mu, log_var = self._shared_infer(batch)

        # gmm part loss
        # only gmm supervision should be low-passed
        if self.lp_mask2d is not None:
            lp_gt_images = self.low_pass_images(gt_images)
        else:
            lp_gt_images = gt_images
        gmm_proj_loss = calc_cor_loss(pred_gmm_images, lp_gt_images, self.mask)
        weighted_gmm_proj_loss = cfg.loss.gmm_cryoem_weight * gmm_proj_loss

        if hasattr(self, "connect_pairs"):
            connect_loss = calc_pair_dist_loss(pred_struc, self.connect_pairs, self.connect_dists)
            weighted_connect_loss = cfg.loss.connect_weight * connect_loss
        else:
            weighted_connect_loss = weighted_gmm_proj_loss.new_tensor(0.)

        if hasattr(self, "sse_pairs"):
            sse_loss = calc_pair_dist_loss(pred_struc, self.sse_pairs, self.sse_dists)
            weighted_sse_loss = cfg.loss.connect_weight * sse_loss
        else:
            weighted_sse_loss = weighted_gmm_proj_loss.new_tensor(0.)

        if hasattr(self, "dist_loss_fn"):
            dist_loss = self.dist_loss_fn(pred_struc)
            # across devices
            all_dist_loss = self.all_gather(dist_loss)  # world_size, batch, num_pairs
            all_dist_loss = all_dist_loss.reshape(-1, dist_loss.shape[-1])

            # chain-wise drop
            with torch.no_grad():
                keep_mask = torch.ones(dist_loss.shape[-1], dtype=torch.bool).to(dist_loss.device)

                for i in range(len(self.cutoff_chain_mask)):
                    tmp_mask = self.cutoff_chain_mask[i]
                    tmp_var = all_dist_loss.index_select(dim=1, index=tmp_mask.nonzero(as_tuple=True)[0]).var(dim=0)
                    intra_chain_keep_mask = tmp_var.lt(torch.quantile(tmp_var, cfg.loss.dist_keep_ratio))
                    keep_mask[tmp_mask] *= intra_chain_keep_mask

                keep_mask = keep_mask.unsqueeze(0).repeat(dist_loss.size(0), 1)
            dist_loss = torch.mean(dist_loss[keep_mask])

            weighted_dist_loss = cfg.loss.dist_weight * dist_loss

            # dist_penalty = torch.mean(torch.abs(self.dist_loss_fn.get_weights()))
            # weighted_dist_penalty = cfg.loss.dist_penalty_weight * dist_penalty
        else:
            weighted_dist_loss = weighted_gmm_proj_loss.new_tensor(0.)
            # weighted_dist_penalty = weighted_gmm_proj_loss.new_tensor(0.)

        if hasattr(self, "clash_pairs"):
            clash_loss = calc_clash_loss(pred_struc, self.clash_pairs, cfg.loss.clash_min_cutoff)
            weighted_clash_loss = cfg.loss.clash_weight * clash_loss
        else:
            weighted_clash_loss = weighted_gmm_proj_loss.new_tensor(0.)

        # vol loss
        if self.cfg.loss.vol_loss_fn == "rmsf":
            pred = fourier_to_primal_2d(pred_vol_f_images).real
            delta = pred - gt_images
            vol_proj_loss = delta.reshape(bsz, -1).square().mean()
        elif self.cfg.loss.vol_loss_fn == "fmsf":
            f_gt_images = primal_to_fourier_2d(gt_images)
            delta = torch.view_as_real(f_gt_images - pred_vol_f_images)
            delta = delta[einops.repeat(self.image_mask, "ny nx -> b 1 ny nx c", b=delta.shape[0], c=delta.shape[-1])]
            vol_proj_loss = delta.reshape(bsz, -1).square().mean()
        else:
            raise NotImplementedError
        weighted_vol_proj_loss = cfg.loss.vol_cryoem_weight * vol_proj_loss

        # KL
        kl_loss = calc_kl_loss(mu, log_var, cfg.loss.free_bits)
        kl_beta = warmup(cfg.loss.warmup_step, upper=cfg.loss.kl_beta_upper)(self.global_step)
        weighted_kld_loss = kl_beta * kl_loss / self.mask.num_masked

        # clac loss
        loss = (weighted_kld_loss + weighted_gmm_proj_loss + weighted_connect_loss + weighted_dist_loss +
                weighted_sse_loss + weighted_clash_loss + weighted_vol_proj_loss)

        tmp_metric = {
            "loss": loss.item(),
            "cryoem(gmm)": weighted_gmm_proj_loss.item(),
            "cryoem(vol)": weighted_vol_proj_loss.item(),
            "con": weighted_connect_loss.item(),
            "sse": weighted_sse_loss.item(),
            "dist": weighted_dist_loss.item(),
            # "dist_penalty": weighted_dist_penalty.item(),
            "clash": weighted_clash_loss.item(),
            "kld": weighted_kld_loss.item(),
            "kld(/dim)": kl_loss.item()
        }
        if self.global_step % cfg.runner.log_every_n_step == 0:
            self.log_dict(tmp_metric)
            log_to_current(f"epoch {self.current_epoch} [{batch_idx}/{self.trainer.num_training_batches}] | " +
                           pretty_dict(tmp_metric, 5))
        return loss

    def validation_step(self, batch, batch_idx):
        gt_images = batch["proj"]
        idxes = batch["idx"]

        # if self.lp_mask2d is not None:
        #     gt_images = self.low_pass_images(gt_images)

        mu, log_var = self.model.encode(prepare_images(gt_images, self.cfg.model.input_space), idxes)
        z = mu
        self.validation_step_outputs.append({"z": z, "idx": idxes})

    def on_validation_epoch_end(self):
        # lightning will automatically copy val samples to let val_loader to be divided by gpu_num with no remainder,
        # here use sample id to remove redundancy
        all_outputs = merge_step_outputs(self.validation_step_outputs)
        all_outputs = self.all_gather(all_outputs)
        all_outputs = squeeze_dict_outputs_1st_dim(all_outputs)

        if self.trainer.is_global_zero and len(all_outputs) > 0:
            # save projection images for checking
            self._shared_image_check()

            save_dir = self._get_save_dir()

            # --------
            # dealing with all z
            indices = get_1st_unique_indices(all_outputs["idx"])
            log_to_current(f"Total {len(indices)} unique samples")
            all_outputs = filter_outputs_by_indices(all_outputs, indices)
            z_list = all_outputs["z"]
            z_list = z_list.cpu().numpy()  # (num_samples, latent_dim)

            np.save(osp.join(save_dir, "z.npy"), z_list)

            # --------
            # Kmeans cluster
            kmeans_labels, centers = cluster_kmeans(z_list, self.cfg.analyze.cluster_k)
            centers, centers_ind = get_nearest_point(z_list, centers)

            if z_list.shape[-1] > 2 and not self.cfg.analyze.skip_umap:
                log_to_current("Running UMAP...")
                z_emb, reducer = run_umap(z_list)
                centers_emb = reducer.transform(centers)
                try:
                    plot_z_dist(z_emb, extra_cluster=centers_emb, save_path=osp.join(save_dir, "z_distribution.png"))
                except Exception as e:
                    log_to_current(e)

            elif z_list.shape[-1] <= 2:
                try:
                    plot_z_dist(z_list, extra_cluster=centers, save_path=osp.join(save_dir, "z_distribution.png"))
                except Exception as e:
                    log_to_current(e)
            else:
                #
                pass

            centers = torch.from_numpy(centers)
            pred_struc, pred_vol = self._shared_decoding(centers)

            self._save_batched_strucs(pred_struc, f"{save_dir}/pred.pdb")

            if self.global_step > (self.trainer.max_steps // 2):
                mkdir_or_exist(osp.join(save_dir, "kmeans"))
                self._save_batched_vols(pred_vol, osp.join(save_dir, "kmeans"))

            # --------
            # pca
            for pca_dim in range(1, 1 + min(3, self.cfg.model.model_cfg.latent_dim)):
                pc, pca = run_pca(z_list)
                start = np.percentile(pc[:, pca_dim - 1], 5)
                stop = np.percentile(pc[:, pca_dim - 1], 95)
                z_pc_traj = get_pc_traj(pca, z_list.shape[1], 10, pca_dim, start, stop)
                z_pc_traj, _ = get_nearest_point(z_list, z_pc_traj)

                z_pc_traj = torch.from_numpy(z_pc_traj)
                pred_struc, pred_vol = self._shared_decoding(z_pc_traj)

                self._save_batched_strucs(pred_struc, f"{save_dir}/pca-{pca_dim}.pdb")
                if self.global_step > (self.trainer.max_steps // 2):
                    mkdir_or_exist(osp.join(save_dir, f"pca_{pca_dim}"))
                    self._save_batched_vols(pred_vol, osp.join(save_dir, f"pca_{pca_dim}"))

        # important
        self.trainer.strategy.barrier()
        self.validation_step_outputs.clear()  # free memory

    def configure_optimizers(self):
        params = [*self.model.parameters()]
        if self.cfg.gmm.tunable:
            params.extend([self.gmm_sigmas, self.gmm_amps])
        if hasattr(self, "dist_loss_fn"):
            params.extend(self.dist_loss_fn.parameters())
        optimizer = optim.AdamW(params, lr=self.cfg.optimizer.lr)
        return optimizer

    # extra hooks:
    # here self.device is set to cuda:0, 1 etc
    def on_fit_start(self):
        self.cfg.work_dir = self.trainer.strategy.broadcast(self.cfg.work_dir)

        # make sure model parameters are the same
        log_to_current(f"load rank 0 model weights")
        state_dict = self.trainer.strategy.broadcast(self.model.state_dict())
        self.model.load_state_dict(state_dict)

    def on_validation_start(self):
        log_to_current(f"Epoch {self.current_epoch} Step {self.global_step} start validation")
        state_dict = self.trainer.strategy.broadcast(self.model.state_dict())
        self.model.load_state_dict(state_dict)

        if self.trainer.is_global_zero:
            save_dir = self._get_save_dir()
            self._save_ckpt(osp.join(save_dir, "ckpt.pt"))
            self.history_saved_dirs.append(save_dir)
            keep_last_k = 1
            if len(self.history_saved_dirs) >= keep_last_k:
                for to_remove in self.history_saved_dirs[:-keep_last_k]:
                    p = Path(to_remove) / "ckpt.pt"
                    if p.exists():
                        p.unlink()
                        log_to_current(f"delete {p} to keep last {keep_last_k} ckpts")

    def on_train_start(self):
        self.trainer.validate_loop.setup_data()
        if self.trainer.is_global_zero:
            self._shared_image_check()
        self.trainer.strategy.barrier()


def train():
    cfg = pl_init_exp(exp_prefix=TASK_NAME, backup_list=[
        __file__,
    ], inplace=False)

    if cfg.seed is not None:
        set_seed(cfg.seed)
        log_to_current(f"seed set to {cfg.seed}")

    dataset = StarfileDataSet(
        StarfileDatasetConfig(
            dataset_dir=cfg.dataset_attr.dataset_dir,
            starfile_path=cfg.dataset_attr.starfile_path,
            apix=cfg.dataset_attr.apix,
            side_shape=cfg.dataset_attr.side_shape,
            down_side_shape=cfg.data_process.down_side_shape,
            mask_rad=cfg.data_process.mask_rad,
            power_images=1.0,
            ignore_rots=False,
            ignore_trans=False, ))

    #
    if cfg.dataset_attr.apix is None:
        cfg.dataset_attr.apix = dataset.apix
    if cfg.dataset_attr.side_shape is None:
        cfg.dataset_attr.side_shape = dataset.side_shape
    if cfg.data_process.down_side_shape is None:
        if dataset.side_shape > 256:
            cfg.data_process.down_side_shape = 128
            dataset.down_side_shape = 128
        else:
            cfg.data_process.down_side_shape = dataset.down_side_shape

    cfg.data_process["down_apix"] = dataset.apix
    if dataset.down_side_shape != dataset.side_shape:
        cfg.data_process.down_apix = dataset.side_shape * dataset.apix / dataset.down_side_shape

    rank_zero_only(cfg.dump)(osp.join(cfg.work_dir, "config.py"))

    log_to_current(f"Load dataset from {dataset.cfg.dataset_dir}, power scaled by {dataset.cfg.power_images}")
    log_to_current(f"Total {len(dataset)} samples")
    log_to_current(f"The dataset side_shape: {dataset.side_shape}, apix: {dataset.apix}")
    log_to_current(f"Set down-sample side_shape {dataset.down_side_shape} with apix {cfg.data_process.down_apix}")

    train_loader = DataLoader(dataset,
                              batch_size=cfg.data_loader.train_batch_per_gpu,
                              shuffle=True,
                              drop_last=True,
                              num_workers=cfg.data_loader.workers_per_gpu)

    test_loader = DataLoader(dataset,
                             batch_size=cfg.data_loader.val_batch_per_gpu,
                             shuffle=False,
                             drop_last=False,
                             num_workers=cfg.data_loader.workers_per_gpu)

    em_task = CryoEMTask(cfg, dataset)

    if not cfg.eval_mode and cfg.do_ref_init:
        init_task = InitTask(em_task)
        # if you meet libibverbs warnings, try process_group_backend="gloo"
        init_trainer = pl.Trainer(max_epochs=3,
                                  devices=cfg.trainer.devices,
                                  accelerator="gpu" if torch.cuda.is_available() else "cpu",
                                  precision=cfg.trainer.precision,
                                  strategy=DDPStrategy(process_group_backend="nccl", find_unused_parameters=True),
                                  logger=False,
                                  enable_checkpointing=False,
                                  enable_model_summary=False,
                                  enable_progress_bar=False,
                                  num_sanity_val_steps=0)

        init_trainer.fit(init_task, train_dataloaders=train_loader)

    em_trainer = pl.Trainer(accelerator="gpu" if torch.cuda.is_available() else "cpu",
                            strategy=DDPStrategy(process_group_backend="nccl"),
                            logger=False,
                            enable_checkpointing=False,
                            enable_model_summary=False,
                            enable_progress_bar=False,
                            **cfg.trainer)

    if not cfg.eval_mode:
        em_trainer.fit(model=em_task, train_dataloaders=train_loader, val_dataloaders=test_loader)
    else:
        em_trainer.validate(model=em_task, dataloaders=test_loader)


if __name__ == "__main__":
    train()
