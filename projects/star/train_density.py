import os
import os.path as osp

import einops
import lightning.pytorch as pl
import numpy as np
import torch
from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch.utilities import rank_zero_only
from torch.utils.data import DataLoader
from tqdm import tqdm
from mmengine import mkdir_or_exist

from cryostar.utils.dataio import StarfileDataSet, StarfileDatasetConfig
from cryostar.nerf.volume_utils import ImplicitFourierVolume
from cryostar.utils.transforms import SpatialGridTranslate
from cryostar.utils.ctf_utils import CTFRelion
from cryostar.utils.fft_utils import (fourier_to_primal_2d, primal_to_fourier_2d)
from cryostar.utils.latent_space_utils import sample_along_pca, get_nearest_point, cluster_kmeans
from cryostar.utils.misc import (pl_init_exp, create_circular_mask, log_to_current, pretty_dict)
from cryostar.utils.losses import calc_kl_loss
from cryostar.utils.ml_modules import VAEEncoder, reparameterize
from cryostar.utils.mrc_tools import save_mrc

from miscs import infer_ctf_params_from_config

log_to_current = rank_zero_only(log_to_current)

TASK_NAME = "density"


class CryoModel(pl.LightningModule):

    def __init__(self, cfg, dataset):
        super().__init__()
        self.cfg = cfg
        self.dataset = dataset
        self.z_dim = cfg.model.z_dim
        self.history_saved_dirs = []
        if cfg.extra_input_data_attr.given_z is None and self.z_dim != 0:
            if cfg.model.enc_space == "real":
                self.encoder = VAEEncoder(self.cfg.data_process.down_side_shape**2,
                                          cfg.model.hidden,
                                          self.z_dim,
                                          num_hidden_layers=4)
            elif cfg.model.enc_space == "fourier":
                self.encoder = VAEEncoder(2 * self.cfg.data_process.down_side_shape**2,
                                          cfg.model.hidden,
                                          self.z_dim,
                                          num_hidden_layers=4)
            else:
                raise NotImplementedError
        self.translate = SpatialGridTranslate(self.cfg.data_process.down_side_shape, )

        ctf_params = infer_ctf_params_from_config(cfg)
        self.ctf = CTFRelion(**ctf_params, num_particles=len(dataset))
        log_to_current(ctf_params)

        self.vol = ImplicitFourierVolume(
            self.z_dim, self.cfg.data_process.down_side_shape, self.cfg.loss.mask_rad_for_image_loss, {
                "net_type": cfg.model.net_type,
                "pe_dim": self.cfg.data_process.down_side_shape,
                "D": self.cfg.data_process.down_side_shape,
                "pe_type": cfg.model.pe_type,
                "force_symmetry": False,
                "hidden": cfg.model.hidden,
            })
        mask = create_circular_mask(self.cfg.data_process.down_side_shape, self.cfg.data_process.down_side_shape, None,
                                    self.cfg.data_process.down_side_shape // 2 * self.cfg.loss.mask_rad_for_image_loss,)
        self.register_buffer("mask", torch.from_numpy(mask))
        if cfg.extra_input_data_attr.given_z is not None:
            self.register_buffer("given_z", torch.from_numpy(np.load(cfg.extra_input_data_attr.given_z)))

        if getattr(self.cfg.extra_input_data_attr, "ckpt_path", None) is not None:
            log_to_current(f"load checkpoint from {self.cfg.extra_input_data_attr.ckpt_path}")
            state_dict = torch.load(self.cfg.extra_input_data_attr.ckpt_path, map_location=self.device)
            self.vol.load_state_dict(state_dict)

    def _get_save_dir(self):
        save_dir = os.path.join(self.cfg.work_dir, f"{self.current_epoch:04d}_{self.global_step:07d}")
        mkdir_or_exist(save_dir)
        return save_dir

    def process_image(self, batch):
        R = batch["rotmat"]
        bsz = len(R)
        trans = torch.cat([
            batch["shiftY"].float().reshape(bsz, 1, 1) / self.cfg.data_process.down_apix,
            batch["shiftX"].float().reshape(bsz, 1, 1) / self.cfg.data_process.down_apix
        ],
                          dim=2)
        proj_in = batch["proj"].to(self.device)
        proj = self.translate.transform(proj_in.squeeze(1), trans.to(self.device))
        if self.cfg.model.shift_data:
            return proj, proj
        else:
            return proj_in, proj

    def training_step(self, batch, batch_idx):
        R = batch["rotmat"]
        bsz = len(R)
        proj_in, proj_out = self.process_image(batch)
        f_proj_in = primal_to_fourier_2d(proj_in)

        if self.z_dim != 0:
            if self.cfg.extra_input_data_attr.given_z is not None:
                z = self.given_z[batch["idx"]].reshape(bsz, -1)
                kld_loss = 0.0
            else:
                if self.cfg.model.enc_space == "fourier":
                    enc_input = einops.rearrange(torch.view_as_real(f_proj_in), "b 1 ny nx c2 -> b (1 ny nx c2)", c2=2)
                elif self.cfg.model.enc_space == "real":
                    enc_input = einops.rearrange(proj_in, "b 1 ny nx -> b (1 ny nx)")
                mu, log_var = self.encoder(enc_input)
                z = reparameterize(mu, log_var)
                kld_loss = calc_kl_loss(mu, log_var, self.cfg.loss.free_bits)
                kld_loss = kld_loss / self.mask.sum()
            f_pred = self.vol(z, R)
        else:
            f_pred = self.vol(None, R)

        pred_ctf_params = {k: batch[k] for k in ('defocusU', 'defocusV', 'angleAstigmatism') if k in batch}
        f_pred = self.ctf(f_pred, batch['idx'], ctf_params=pred_ctf_params, mode="gt", frequency_marcher=None)

        if self.cfg.loss.loss_fn == "rmsf":
            pred = fourier_to_primal_2d(f_pred).real
            delta = pred - proj_out
            em_loss = delta.reshape(bsz, -1).square().mean()
        elif self.cfg.loss.loss_fn == "fmsf":
            f_proj = primal_to_fourier_2d(proj_out)
            delta = torch.view_as_real(f_proj - f_pred)
            delta = delta[einops.repeat(self.mask, "ny nx -> b 1 ny nx c", b=delta.shape[0], c=delta.shape[-1])]
            em_loss = delta.reshape(bsz, -1).square().mean()
        else:
            raise NotImplementedError

        loss = em_loss
        log_dict = {"em": em_loss}
        if self.z_dim != 0:
            log_dict["kld"] = kld_loss
            loss = loss + kld_loss
        if self.global_step % 100 == 0:
            log_to_current(f"epoch {self.current_epoch} [{batch_idx}/{self.trainer.num_training_batches}] | " +
                           pretty_dict(log_dict, 5))
        return loss

    def on_validation_start(self) -> None:
        self.evaluate()

    def validation_step(self, *args, **kwargs):
        pass

    def save_ckpt(self):
        if self.trainer.is_global_zero:
            save_dir = self._get_save_dir()
            torch.save(self.vol.state_dict(), os.path.join(save_dir, "ckpt.pt"))
            # self.history_saved_dirs.append(save_dir)
            # keep_last_k = 1
            # if len(self.history_saved_dirs) >= keep_last_k:
            #     for to_remove in self.history_saved_dirs[:-keep_last_k]:
            #         p = Path(to_remove) / "ckpt.pt"
            #         if p.exists():
            #             p.unlink()
            #             log_to_current(f"delete {p} to keep last {keep_last_k} ckpts")

    def evaluate(self) -> None:
        pixel_size = self.cfg.data_process.down_apix
        valid_loader = DataLoader(dataset=self.dataset,
                                  batch_size=self.cfg.data_loader.val_batch_per_gpu,
                                  shuffle=False,
                                  drop_last=False,
                                  num_workers=12)
        if self.trainer.is_global_zero:
            save_dir = self._get_save_dir()
            self.save_ckpt()
            if self.z_dim != 0:
                if self.cfg.extra_input_data_attr.given_z is None:
                    zs = []
                    for batch in tqdm(iter(valid_loader)):
                        proj_in, proj_out = self.process_image(batch)
                        f_proj_in = primal_to_fourier_2d(proj_in)
                        if self.cfg.model.enc_space == "real":
                            enc_input = einops.rearrange(proj_in, "b 1 ny nx -> b (1 ny nx)").to(self.device)
                        else:
                            enc_input = einops.rearrange(torch.view_as_real(f_proj_in),
                                                         "b 1 ny nx c2 -> b (1 ny nx c2)",
                                                         c2=2).to(self.device)
                        mu, log_var = self.encoder(enc_input)
                        zs.append(mu.detach().cpu())
                        # if self.cfg.trainer.devices == 1 and len(zs) > 20:
                        #     for _ in range(10):
                        #         log_to_current("WARNING!" + "*" * _)
                        #     log_to_current(
                        #         "since only one device is used, we assume this is a debug mode, and do not go through all validsets"
                        #     )
                        #     break
                    zs = torch.cat(zs).cpu().numpy()
                else:
                    zs = self.given_z.cpu().numpy()
                np.save(f"{save_dir}/z.npy", zs)

                kmeans_labels, centers = cluster_kmeans(zs, 10)
                centers, centers_ind = get_nearest_point(zs, centers)
                np.savetxt(f"{save_dir}/z_kmeans.txt", centers, fmt='%.5f')
                np.savetxt(f"{save_dir}/z_kmeans_ind.txt", centers_ind, fmt='%d')
                centers = torch.from_numpy(centers).to(self.device)
                for i in range(len(centers)):
                    v = self.vol.make_volume(centers[i:i + 1])
                    save_mrc(v.cpu().numpy(), f"{save_dir}/vol_kmeans_{i:03}.mrc", pixel_size,
                             -pixel_size * (v.shape[0] // 2))

                for pca_dim in range(1, 1 + min(3, self.cfg.model.z_dim)):
                    z_on_pca, z_on_pca_id = sample_along_pca(zs, pca_dim, 10)
                    np.savetxt(f"{save_dir}/z_pca_{pca_dim}.txt", z_on_pca, fmt='%.5f')
                    np.savetxt(f"{save_dir}/z_pca_ind_{pca_dim}.txt", z_on_pca_id, fmt='%d')
                    z_on_pca = torch.from_numpy(z_on_pca).to(self.device)
                    for i in range(len(z_on_pca)):
                        v = self.vol.make_volume(z_on_pca[i:i + 1])
                        save_mrc(v.cpu().numpy(), f"{save_dir}/vol_pca_{pca_dim}_{i:03}.mrc", pixel_size,
                                 -pixel_size * (v.shape[0] // 2))
            else:
                v = self.vol.make_volume(None)
                save_mrc(v.cpu().numpy(), f"{save_dir}/vol.mrc", pixel_size, -pixel_size * (v.shape[0] // 2))

    def on_train_start(self) -> None:
        if self.trainer.is_global_zero:
            log_to_current(self)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), 0.0001)


def train():
    cfg = pl_init_exp(exp_prefix=TASK_NAME, backup_list=[
        __file__,
    ], inplace=False)

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
            cfg.data_process.down_side_shape = 256
            dataset.down_side_shape = 256
        else:
            cfg.data_process.down_side_shape = dataset.down_side_shape

    cfg.data_process["down_apix"] = dataset.apix
    if dataset.down_side_shape != dataset.side_shape:
        cfg.data_process.down_apix = dataset.side_shape * dataset.apix / dataset.down_side_shape

    rank_zero_only(cfg.dump)(osp.join(cfg.work_dir, "config.py"))

    if cfg.dataset_attr.f_mu is not None:
        dataset.f_mu = cfg.dataset_attr.f_mu
        dataset.f_std = cfg.dataset_attr.f_std
    else:
        dataset.estimate_normalization()
    log_to_current(f"Fourier mu/std: {dataset.f_mu:.5f}/{dataset.f_std:.5f}")

    cryo_model = CryoModel(cfg, dataset)
    train_loader = DataLoader(dataset=dataset,
                              batch_size=cfg.data_loader.train_batch_per_gpu,
                              shuffle=True,
                              num_workers=cfg.data_loader.workers_per_gpu)

    trainer = pl.Trainer(accelerator="gpu" if torch.cuda.is_available() else "cpu",
                         strategy=DDPStrategy(find_unused_parameters=True),
                         logger=False,
                         enable_checkpointing=False,
                         enable_model_summary=False,
                         enable_progress_bar=False,
                         **cfg.trainer)

    if not cfg.eval_mode:
        trainer.fit(cryo_model, train_dataloaders=train_loader, val_dataloaders=["DUMMY VALID LOADER"])
    else:
        trainer.validate(model=cryo_model, dataloaders=["DUMMY VALID LOADER"])


if __name__ == "__main__":
    train()
