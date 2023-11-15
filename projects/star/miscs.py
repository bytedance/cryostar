from pathlib import Path

import starfile
import cv2
import numpy as np
from scipy.spatial import distance
import biotite.structure as struc
from PIL import Image
from torchvision.transforms import ToPILImage
from torchvision.utils import make_grid
import seaborn as sns
from matplotlib import pyplot as plt

from functools import lru_cache
import einops

try:
    import cupy as cp  # type: ignore
except ImportError:
    cp = np

import torch
from torch import linalg as LA
from torch import nn
import torch.nn.functional as F

from cryostar.common.residue_constants import ca_ca
from cryostar.utils.rotation_conversion import matrix_to_euler_angles, axis_angle_to_matrix
from cryostar.utils.pdb_tools import bt_read_pdb, bt_save_pdb
from cryostar.utils.polymer import get_num_electrons
from cryostar.utils.misc import log_to_current, ASSERT_SHAPE
from cryostar.utils.latent_space_utils import run_umap, run_pca, get_pc_traj, get_nearest_point, cluster_kmeans # noqa
from cryostar.utils.ml_modules import VAEEncoder, Decoder, reparameterize
from cryostar.utils.ctf import parse_ctf_star

from lightning.pytorch.utilities import rank_zero_only
from typing import Union
from abc import abstractmethod, ABC


CA_CA = round(ca_ca, 2)
log_to_current = rank_zero_only(log_to_current)


def infer_ctf_params_from_config(cfg):
    star_file_path = Path(cfg.dataset_attr.starfile_path)
    ctf_params = parse_ctf_star(star_file_path, side_shape=cfg.data_process.down_side_shape,
                                apix=cfg.data_process.down_apix)[0].tolist()
    ctf_params = {
        "size": int(ctf_params[0]),
        "resolution": ctf_params[1],
        "kV": ctf_params[5],
        "cs": ctf_params[6],
        "amplitudeContrast": ctf_params[7]
    }
    return ctf_params


def annealer(period, sleep=0, lower=0.0, upper=1.0):
    """
        return a function f: x->y which looks like:
        ```
               /----|   /----|
              /     |  /     |
             /      | /      |
        ----/       |/       |
        ```
        1. sleep
        2. increase for half period
        3. sleep for half period
    """
    half_period = period // 2
    value_range = (upper - lower)

    def run(cur_step):
        if cur_step < sleep:
            return lower
        in_period_step = (cur_step - sleep) % period
        if in_period_step < half_period:
            return in_period_step / half_period * value_range
        else:
            return upper

    return run


def warmup(warmup_step, lower=0.0, upper=1.0):
    value_range = (upper - lower)

    def run(cur_step):
        if cur_step < warmup_step:
            return (cur_step / warmup_step) * value_range
        else:
            return upper

    return run


def save_tensor_image(tensors, save_path, mask=None):
    # normalize
    max_val = torch.max(tensors.flatten(start_dim=1), 1)[0][:, None, None, None]
    min_val = torch.min(tensors.flatten(start_dim=1), 1)[0][:, None, None, None]
    tensors = (tensors - min_val) / (max_val - min_val)

    show_img = ToPILImage()(make_grid(tensors, nrow=5))
    if mask is None:
        show_img.save(save_path)
    else:
        show_img = np.copy(np.asarray(show_img))
        # show_img = cv2.cvtColor(show_img, cv2.COLOR_GRAY2RGB)
        if mask.ndim == 2:
            mask = mask[None]
        mask = ToPILImage()(make_grid(mask.expand(tensors.shape[0], -1, -1, -1), nrow=5))
        mask = np.invert(np.asarray(mask).astype(bool))[..., 0]
        color_mask = np.array([[0, 0, 0], [31, 119, 180]], dtype=np.uint8)
        color_mask = color_mask[mask.astype(int)]
        show_img[mask] = cv2.addWeighted(show_img[mask], 0.5, color_mask[mask], 0.5, 0)
        show_img = Image.fromarray(show_img)
        show_img.save(save_path)


def plt_twin(x,
             y1,
             y2,
             x_label,
             y1_label,
             y2_label,
             y1_lim=None,
             y2_lim=None,
             y1_color='tab:red',
             y2_color='tab:blue',
             save_path=None):
    # assume x is continuous, fill empty y with np.nan
    plt.clf()
    total_num = max(x) - min(x) + 1
    new_x = np.arange(min(x), max(x) + 1)
    new_y1 = np.empty(total_num, dtype=y1.dtype)
    new_y1.fill(np.nan)
    new_y2 = np.empty(total_num, dtype=y2.dtype)
    new_y2.fill(np.nan)

    new_y1[x - min(x)] = y1
    new_y2[x - min(x)] = y2

    fig, ax = plt.subplots()
    ax.plot(new_x, new_y1, color=y1_color)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y1_label)
    if y1_lim is not None:
        ax.set_ylim(y1_lim[0], y1_lim[1])
    ax.yaxis.label.set_color(y1_color)
    ax.tick_params(axis='y', colors=y1_color)

    ax2 = ax.twinx()
    ax2.plot(new_x, new_y2, color=y2_color)
    ax2.set_ylabel(y2_label)
    if y2_lim is not None:
        ax2.set_ylim(y2_lim[0], y2_lim[1])
    ax.yaxis.label.set_color(y2_color)
    ax.tick_params(axis='y', colors=y2_color)

    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()


def plt_single(x, y, x_label, y_label, y_color='tab:blue', x_lim=None, y_lim=None, save_path=None):
    # assume x is continuous, fill empty y with np.nan
    total_num = max(x) - min(x) + 1
    new_x = np.arange(min(x), max(x) + 1)
    new_y = np.empty(total_num, dtype=y.dtype)
    new_y.fill(np.nan)

    new_y[x - min(x)] = y

    fig, ax = plt.subplots()
    ax.plot(new_x, new_y, color=y_color)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if x_lim is not None:
        ax.set_xlim(x_lim)
    if y_lim is not None:
        ax.set_ylim(y_lim)

    # plot max and min value
    id_arr = np.argsort(y)
    ax.plot(x[id_arr[0]], y[id_arr[0]], '.', color='tab:orange')
    ax.text(x[id_arr[0]], y[id_arr[0]], f"{y[id_arr[0]]:.4f}")

    ax.plot(x[id_arr[-1]], y[id_arr[-1]], '.', color='tab:orange')
    ax.text(x[id_arr[-1]], y[id_arr[-1]], f"{y[id_arr[-1]]:.4f}")

    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()


def plt_z(x, y, x_lim=None, y_lim=None, save_path=None):
    # plot z scatter
    plt.clf()
    plt.scatter(x, y, marker='.', s=2)

    u = np.unique(x)
    median_val_list = []
    for i in u:
        median_val_list.append(np.median(y[x == i]))

    plt.scatter(u, median_val_list, marker='.', color='tab:orange')
    plt.xlabel("true reaction coordinate")
    plt.ylabel("latent encoding")
    if x_lim is not None:
        plt.xlim(x_lim)
    if y_lim is not None:
        plt.ylim(y_lim)

    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()


def plot_z_dist(z, extra_cluster=None, save_path=None):
    if z.shape[-1] == 1:
        fig = sns.displot(x=z[:, 0])
        fig.set_xlabels("z values")
        if save_path is not None:
            fig.savefig(save_path)
    elif z.shape[-1] == 2:
        sns.set()
        fig = sns.jointplot(x=z[:, 0], y=z[:, 1], kind="kde", fill=True)
        ax = fig.figure.axes
        if extra_cluster is not None:
            ax[0].scatter(extra_cluster[:, 0], extra_cluster[:, 1], marker='.', color='tab:orange')
        if save_path is not None:
            fig.savefig(save_path)
    else:
        raise ValueError(f"input z with shape {z.shape}")


def get_mse(ref, obj):
    return np.mean(np.power(ref - obj, 2))


def get_nmse(ref, obj):
    # return get_mse(ref, obj) / get_mse(ref, np.mean(ref))
    return get_mse(ref, obj) / get_mse(ref, 0)


def low_pass_mask3d(shape, apix=1., bandwidth=2):
    freq = np.fft.fftshift(np.fft.fftfreq(shape, apix))
    freq = freq**2
    freq = np.sqrt(freq[:, None, None] + freq[None, :, None] + freq[None, None])

    mask = np.asarray(freq < 1 / bandwidth, dtype=np.float32)
    # trick to avoid "ringing", however you should increase sigma to about 11 to completely remove artifact
    # gaussian_filter(mask, 3, output=mask)
    return mask


def low_pass_mask2d(shape, apix=1., bandwidth=2):
    freq = np.fft.fftshift(np.fft.fftfreq(shape, apix))
    freq = freq**2
    freq = np.sqrt(freq[:, None] + freq[None, :])

    mask = np.asarray(freq < 1 / bandwidth, dtype=np.float32)
    return mask


# --------
# loss utilities
def get_atom_center(atom_arr):
    return np.mean(atom_arr.coord, axis=0)


def build_gmm(atom_arr, level="ca"):
    level = level.lower()
    assert level in ("ca", "ca+r", "bb+r", "all-atom")
    if level == "ca":
        nums = np.sum(atom_arr.atom_name == "CA")
    elif level == "ca+r":
        nums = np.sum(atom_arr.atom_name == "CA") * 2
    elif level == "bb+r":
        nums = np.sum(atom_arr.atom_name == "CA") * 5  # N, CA, C, O, R-group
    else:
        nums = len(atom_arr)

    meta = {
        "chain_id": np.empty(nums, dtype="U4"),
        "res_id": np.empty(nums, dtype=int),
        "coord": np.empty((nums, 3), dtype=float),
        "num_electron": np.empty(nums, dtype=int),
        "atom_name": np.empty(nums, dtype="U6"),
        "res_name": np.empty(nums, dtype="U3"),
        "element": np.empty(nums, dtype="U2")
    }

    def _set_meta(pos, *vals):
        meta["chain_id"][pos] = vals[0]
        meta["res_id"][pos] = vals[1]
        meta["coord"][pos] = vals[2]
        meta["num_electron"][pos] = vals[3]
        meta["atom_name"][pos] = vals[4]
        meta["res_name"][pos] = vals[5]
        meta["element"][pos] = vals[6]
        return

    pos = 0
    # loop chains
    for chain in struc.chain_iter(atom_arr):
        chain_id = chain.chain_id[0]
        # loop residues
        for res in struc.residue_iter(chain):
            res_id = res.res_id[0]
            res_name = res.res_name[0]

            if level == "ca":
                _set_meta(pos, chain_id, res_id, res[res.atom_name == "CA"].coord, get_num_electrons(res), "CA",
                          res_name, "C")
                pos += 1
            else:
                main_chain = res[np.isin(res.atom_name, ['N', 'CA', 'C', 'O'])]
                side_chain = res[~np.isin(res.atom_name, ['N', 'CA', 'C', 'O'])]

                if level == "ca+r":
                    # CA location as main group
                    _set_meta(pos, chain_id, res_id, res[res.atom_name == "CA"].coord, get_num_electrons(main_chain),
                              "CA", res_name, "C")
                    pos += 1
                    # R group
                    _set_meta(pos, chain_id, res_id, get_atom_center(side_chain), get_num_electrons(side_chain), "R",
                              res_name, "C")
                    pos += 1

                elif level == "bb+r":
                    # N, CA, C, O
                    for atom_name in main_chain.atom_name:
                        tmp_arr = res[res.atom_name == atom_name]
                        _set_meta(pos, chain_id, res_id, tmp_arr.coord, get_num_electrons(tmp_arr), atom_name, res_name,
                                  tmp_arr.element[0])
                        pos += 1
                    # R group
                    _set_meta(pos, chain_id, res_id, get_atom_center(side_chain), get_num_electrons(side_chain), "R",
                              res_name, "C")
                    pos += 1

                else:
                    for atom_name in res.atom_name:
                        tmp_arr = res[res.atom_name == atom_name]
                        _set_meta(pos, chain_id, res_id, tmp_arr.coord, get_num_electrons(tmp_arr), atom_name, res_name,
                                  tmp_arr.element[0])
                        pos += 1

    assert pos == nums
    return meta


def build_pgmm(atom_arr):
    nt_arr = atom_arr[struc.filter_nucleotides(atom_arr)]
    aa_arr = atom_arr[struc.filter_amino_acids(atom_arr)]

    nums = struc.get_residue_count(aa_arr)
    if len(nt_arr) > 0:
        nums += struc.get_residue_count(nt_arr)

    meta = {
        "chain_id": np.empty(nums, dtype="U4"),
        "res_id": np.empty(nums, dtype=int),
        "coord": np.empty((nums, 3), dtype=float),
        "num_electron": np.empty(nums, dtype=int),
        "atom_name": np.empty(nums, dtype="U6"),
        "res_name": np.empty(nums, dtype="U3"),
        "element": np.empty(nums, dtype="U2")
    }

    def _set_meta(pos, *vals):
        meta["chain_id"][pos] = vals[0]
        meta["res_id"][pos] = vals[1]
        meta["coord"][pos] = vals[2]
        meta["num_electron"][pos] = vals[3]
        meta["atom_name"][pos] = vals[4]
        meta["res_name"][pos] = vals[5]
        meta["element"][pos] = vals[6]
        return

    def _update(tmp_arr, kind="aa"):
        nonlocal pos
        for chain in struc.chain_iter(tmp_arr):
            chain_id = chain.chain_id[0]
            for res in struc.residue_iter(chain):
                res_id = res.res_id[0]
                res_name = res.res_name[0]

                if kind == "aa":
                    tmp_res = res[struc.filter_peptide_backbone(res)]
                    _set_meta(pos, chain_id, res_id, tmp_res[tmp_res.atom_name == "CA"].coord, get_num_electrons(res),
                              "CA", res_name, "C")
                elif kind == "nt":
                    if "P" in res.atom_name:
                        _set_meta(pos, chain_id, res_id, res[res.atom_name == "P"].coord, get_num_electrons(res), "P",
                                  res_name, "P")
                    else:
                        print("No P in current NT residue, use C1 instead")
                        _set_meta(pos, chain_id, res_id, res[res.atom_name == "C1'"].coord, get_num_electrons(res),
                                  "C1'", res_name, "C")
                else:
                    raise NotImplemented
                pos += 1

    pos = 0

    _update(aa_arr, kind="aa")
    if len(nt_arr) > 0:
        _update(nt_arr, kind="nt")

    assert pos == nums
    return meta


def template_pdb_from_meta(meta):
    nums = len(meta["coord"])
    atom_arr = struc.AtomArray(nums)
    atom_arr.coord = meta["coord"]

    for k in meta.keys():
        if k != "coord":
            atom_arr.set_annotation(k, meta[k])

    atom_arr.atom_name[atom_arr.atom_name == "R"] = "CB"
    return atom_arr


def valid_dist_pair_from_meta(coord_arr,
                              chain_id_arr,
                              res_id_arr=None,
                              intra_chain_cutoff=15.,
                              inter_chain_cutoff=15.,
                              intra_chain_res_bound=None):
    sel_indices = []
    dist_map = distance.cdist(coord_arr, coord_arr, metric='euclidean')
    # 1. intra chain
    sel_mask = dist_map <= intra_chain_cutoff
    np.fill_diagonal(sel_mask, False)
    # sel_mask = np.triu(sel_mask, k=1)
    # get indices of valid pairs
    indices_in_pdb = np.nonzero(sel_mask)
    indices_in_pdb = np.column_stack((indices_in_pdb[0], indices_in_pdb[1]))
    indices_in_pdb = indices_in_pdb[chain_id_arr[indices_in_pdb[:, 0]] == chain_id_arr[indices_in_pdb[:, 1]]]
    # filter by res_id
    if intra_chain_res_bound is not None:
        assert res_id_arr is not None
        res_ids = res_id_arr[indices_in_pdb]
        res_id_dist = np.abs(np.diff(res_ids, axis=1)).flatten()
        indices_in_pdb = indices_in_pdb[res_id_dist <= intra_chain_res_bound]

    sel_indices.append(indices_in_pdb)

    # 2. inter chain
    if inter_chain_cutoff is not None:
        sel_mask = dist_map <= inter_chain_cutoff
        np.fill_diagonal(sel_mask, False)
        # sel_mask = np.triu(sel_mask, k=1)
        indices_in_pdb = np.nonzero(sel_mask)
        indices_in_pdb = np.column_stack((indices_in_pdb[0], indices_in_pdb[1]))
        indices_in_pdb = indices_in_pdb[chain_id_arr[indices_in_pdb[:, 0]] != chain_id_arr[indices_in_pdb[:, 1]]]
        sel_indices.append(indices_in_pdb)

    sel_indices = np.vstack(sel_indices)

    sel_indices = torch.from_numpy(sel_indices).long()

    centers = torch.from_numpy(coord_arr).float()
    tmp = centers[sel_indices]  # num_pair, 2, 3
    target_dist = LA.vector_norm(torch.diff(tmp, dim=-2), axis=-1).squeeze(-1)  # num_pair
    return sel_indices, target_dist


def possible_clash_pair_from_meta(coord_arr, chain_id_arr, max_cutoff=10., min_cutoff=4.):
    """
    Input:
        min_cutoff: important, set to a meaningful clash threshold.
        max_cutoff: not important, used to filter out residues too far away to avoid computation cost.
    """
    dist_map = distance.cdist(coord_arr, coord_arr, metric='euclidean')
    sel_mask = (dist_map <= max_cutoff) & (dist_map >= min_cutoff)
    indices_in_pdb = np.nonzero(sel_mask)
    indices_in_pdb = np.column_stack((indices_in_pdb[0], indices_in_pdb[1]))
    indices_in_pdb = indices_in_pdb[chain_id_arr[indices_in_pdb[:, 0]] != chain_id_arr[indices_in_pdb[:, 1]]]
    indices_in_pdb = torch.from_numpy(indices_in_pdb).long()
    return indices_in_pdb


def calc_clash_loss(pred_struc, pair_index, clash_cutoff=4.0):
    pred_dist = pred_struc[:, pair_index]  # bsz, num_pair, 2, 3
    pred_dist = LA.vector_norm(torch.diff(pred_dist, dim=-2), axis=-1).squeeze(-1)  # bsz, num_pair
    possible_clash_dist = pred_dist[pred_dist < clash_cutoff]
    if possible_clash_dist.numel() == 0:
        avg_loss = torch.tensor(0.0).to(pred_struc)
    else:
        possible_clash_loss = (clash_cutoff - possible_clash_dist)**2
        avg_loss = possible_clash_loss.mean()
    return avg_loss


def valid_dist_pair(atom_arr, intra_chain_cutoff=15., inter_chain_cutoff=15., intra_chain_res_bound=None):
    return valid_dist_pair_from_meta(atom_arr.coord,
                                     atom_arr.chain_id,
                                     atom_arr.res_id,
                                     intra_chain_cutoff=intra_chain_cutoff,
                                     inter_chain_cutoff=inter_chain_cutoff,
                                     intra_chain_res_bound=intra_chain_res_bound)


def pdb_ca_stat(pdb_path, tol=0.1):
    # get first model
    atom_arr = bt_read_pdb(pdb_path)[0]
    # filter CA
    atom_arr = atom_arr[struc.filter_peptide_backbone(atom_arr)]
    atom_arr = atom_arr[atom_arr.atom_name == "CA"]
    atom_coord = atom_arr.coord

    # filter chain-level CA bonds
    ca_idx_pair = {}
    for chain_id in struc.get_chains(atom_arr):
        chain_mask = atom_arr.chain_id == chain_id
        ca_indices = np.nonzero(chain_mask)[0]
        if len(ca_indices) < 2:
            continue

        # CA bond calc indices pair
        pair_idx = np.column_stack((ca_indices[:-1], ca_indices[1:]))
        dist_mat = atom_coord[pair_idx]
        ca_dist = np.ravel(np.linalg.norm(np.diff(dist_mat, axis=1), axis=-1))
        valid_ca_bond = np.logical_and(ca_dist >= (CA_CA - tol), ca_dist <= (CA_CA + tol))
        valid_pair = pair_idx[valid_ca_bond]
        ca_idx_pair[chain_id] = valid_pair
    return atom_coord, np.vstack(list(ca_idx_pair.values()))


def calc_bond_loss(pred_struc, pair_index=None, tol=None):
    if pair_index is None:
        ca_bonds = LA.vector_norm(pred_struc[:, :-1] - pred_struc[:, 1:], ord=2, dim=2)
    else:
        dist_mat = pred_struc[:, pair_index]  # bsz, num_ca_bond, 2, 3
        ca_bonds = LA.vector_norm(torch.diff(dist_mat, dim=-2), axis=-1).squeeze(-1)  # bsz, num_ca_bond

    bond_loss = torch.abs(ca_bonds - CA_CA)
    if tol is not None:
        bond_loss = torch.maximum(bond_loss - tol, bond_loss.new_tensor(0.))

    bond_loss = torch.square(bond_loss).mean()  # averaged over bsz
    return bond_loss


@lru_cache(maxsize=None)
def prepare_dynamic_loss(
        num_nodes: int,
        pair_index: torch.LongTensor,    # shape: (edge, 2)
        top_p_ratio: float,
    ):
    """
        The left side of pair_index should be sorted in the ascending order!
        [
            [0, _], [0, _], [0, _],
            [1, _], [1, _],
            [2, _], [2, _], [2, _], [2, _],
            ...
        ]
    """
    device = pair_index.device
    num_node_nbrs = [0 for _ in range(num_nodes)]
    left_nodes = pair_index[:, 0].tolist()
    for ele in left_nodes:
        num_node_nbrs[ele] += 1
    num_node_nbrs = torch.tensor(num_node_nbrs, device=device)
    reshape_indices = torch.zeros(num_nodes, max(num_node_nbrs), dtype=torch.long, device=device)
    reshape_valid_mask = torch.zeros(num_nodes, max(num_node_nbrs), dtype=torch.bool, device=device)
    reshape_top_p_mask = torch.zeros(num_nodes, max(num_node_nbrs), dtype=torch.bool, device=device)
    start_idx = 0
    for i in range(num_nodes):
        reshape_indices[i, :num_node_nbrs[i]] = start_idx + torch.arange(num_node_nbrs[i], device=device)
        reshape_valid_mask[i, :num_node_nbrs[i]] = True
        reshape_top_p_mask[i, :int(top_p_ratio * num_node_nbrs[i])] = True
        start_idx += num_node_nbrs[i]
    return num_node_nbrs, reshape_indices, reshape_valid_mask, reshape_top_p_mask


@lru_cache(maxsize=None)
def prepare_dynamic_intra_chain_loss(
    chain_id: tuple,                # shape: (node, ), converted from np.ndarray since it may be unhashable
    pair_index: torch.LongTensor,   # shape: (edge, 2)
):
    chain_id = np.array(chain_id)
    device = pair_index.device
    chain2idx = {}
    idx = 0
    for ele in set(chain_id):
        chain2idx[ele] = idx
        idx += 1

    chain_pairs = [[] for _ in range(len(chain2idx))]
    pair_index_np = pair_index.cpu().numpy()
    pair_chain_id = chain_id[pair_index_np]
    for pair_idx, pair in enumerate(pair_chain_id):
        if pair[0] == pair[1]:
            chain_pairs[chain2idx[pair[0]]].append(pair_idx)
    chain_pairs = [torch.tensor(ele, device=device) for ele in chain_pairs if len(ele) > 10]
    return chain_pairs


def calc_pair_dist_loss(pred_struc, pair_index, target_dist, type="vanilla", chain_id=None):
    bsz = pred_struc.shape[0]
    pred_dist = pred_struc[:, pair_index]  # bsz, num_pair, 2, 3
    pred_dist = LA.vector_norm(torch.diff(pred_dist, dim=-2), axis=-1).squeeze(-1)  # bsz, num_pair
    if type == "vanilla":
        return F.mse_loss(pred_dist, target_dist.repeat(bsz, 1))
    elif "all-var-relax" in type:
        # optional value:
        #   all-var-relax@p0.99    keep bonds whose variance is the smallest 99%
        #   all-var-relax@q1.0     keep bonds whose variance >= 1.0
        if "@" in type:
            arg = type.split("@")[1]
            assert arg[0] in ["p", "q"]
            use_percentile = arg[0] == "p"
            loss_filter = float(arg[1:])
        else:
            use_percentile = True
            loss_filter = 0.99
        loss = F.mse_loss(pred_dist, target_dist.repeat(bsz, 1), reduction="none")
        loss_var = loss.var(0, keepdim=False).detach()
        # if "var-relax-ema" in type:
        #     other.running_variance = 0.9 * other.running_variance + 0.1 * loss_var
        #     loss_var = other.running_variance
        if np.random.rand() < 0.001:
            log_to_current("variance statistics:")
            q = [0.0, 0.9, 0.95, 0.97, 0.99, 0.999]
            v = torch.quantile(loss_var, torch.tensor(q, device=loss.device)).tolist()
            log_to_current("|".join([f" {q[i] * 100}%: {v[i]:.3f} " for i in range(len(q))]))
            p = [0.25, 1.0, 4.0, 16.0]
            v = [(loss_var > p[i]).sum() / len(loss_var) for i in range(len(p))]
            log_to_current("|".join([f" {p[i]}: {v[i] * 100:.1f}% " for i in range(len(p))]))
        if use_percentile:
            loss_ind = loss_var.sort(descending=False).indices
            loss = loss.index_select(1, loss_ind[:int(len(loss_var) * loss_filter)])
        else:
            loss_mask = loss_var < loss_filter
            loss = loss[loss_mask[None, :].repeat(bsz, 1)]
        avg_loss = loss.mean()
        return avg_loss
    elif "chain-var-relax" in type:
        if "@" in type:
            arg = type.split("@")[1]
            loss_filter = float(arg[1:])
        else:
            loss_filter = 0.95
        loss = F.mse_loss(pred_dist, target_dist.repeat(bsz, 1), reduction="none")
        chain_pairs = prepare_dynamic_intra_chain_loss(tuple(chain_id), pair_index)
        chain_losses = []
        for i in range(len(chain_pairs)):
            chain_loss = loss.index_select(1, chain_pairs[i])
            chain_loss_var = chain_loss.var(0, keepdim=False).detach()
            chain_loss_ind = chain_loss_var.sort(descending=False).indices
            chain_loss = chain_loss.index_select(1, chain_loss_ind[:int(len(chain_loss_var) * loss_filter)])
            chain_losses.append(chain_loss)
        loss = torch.cat(chain_losses, 1)
        avg_loss = loss.mean()
        return avg_loss
    elif type == "inverse":
        target_dist = target_dist.repeat(bsz, 1)
        loss = F.mse_loss(pred_dist, target_dist, reduction="none")
        lt6_loss = (loss[target_dist <= 6]).sum()
        gt6_loss = loss[target_dist > 6]
        gt6_weight = 1 / (target_dist[target_dist > 6].detach() - 5)
        gt6_loss = (gt6_loss * gt6_weight).sum()
        total_loss = lt6_loss + gt6_loss
        avg_loss = total_loss / target_dist.numel()
        return avg_loss
    elif "dynamic" in type:
        if "@" in type:
            ratio = float(type.split("@")[1])
        else:
            ratio = 0.85
        num_nodes = pred_struc.shape[1]
        num_node_nbrs, reshape_indices, reshape_valid_mask, reshape_top_p_mask = prepare_dynamic_loss(
            num_nodes, pair_index, ratio)
        dist_mse = (pred_dist - target_dist)**2  # bsz x num_nodes
        dist_mse = dist_mse.index_select(1, reshape_indices.reshape(-1))  # bsz x (num_nodes * max_node_nbr)
        dist_mse = dist_mse.reshape(bsz, num_nodes, num_node_nbrs.max())
        dist_mse = dist_mse.masked_fill(~reshape_valid_mask[None, ...], 10000.)
        dist_mse = dist_mse.sort(descending=False, dim=2).values  # bsz x num_nodes x max_node_nbr
        batch_mask = einops.repeat(reshape_top_p_mask, "num_nodes max_node_nbr -> bsz num_nodes max_node_nbr", bsz=bsz)
        avg_loss = dist_mse[batch_mask].sum() / batch_mask.sum()
        return avg_loss
    elif type == "90p":
        target_dist = target_dist.repeat(bsz, 1)
        loss = F.mse_loss(pred_dist, target_dist, reduction="none")
        mask = torch.le(loss, torch.quantile(loss, 0.9, dim=1, keepdim=True))
        avg_loss = loss[mask].sum() / mask.float().sum()
        return avg_loss
    else:
        raise NotImplementedError


# --------
# pl helper
def merge_step_outputs(outputs):
    ks = outputs[0].keys()
    res = {}
    for k in ks:
        res[k] = torch.concat([out[k] for out in outputs], dim=0)
    return res


def squeeze_dict_outputs_1st_dim(outputs):
    res = {}
    for k in outputs.keys():
        res[k] = outputs[k].flatten(start_dim=0, end_dim=1)
    return res


def filter_outputs_by_indices(outputs, indices):
    res = {}
    for k in outputs.keys():
        res[k] = outputs[k][indices]
    return res


def get_1st_unique_indices(t):
    _, idx, counts = torch.unique(t, dim=None, sorted=True, return_inverse=True, return_counts=True)
    # ind_sorted: the index corresponding to same unique value will be grouped by these indices
    _, ind_sorted = torch.sort(idx, stable=True)
    cum_sum = counts.cumsum(0)
    cum_sum = torch.cat((cum_sum.new_tensor([
        0,
    ]), cum_sum[:-1]))
    first_idx = ind_sorted[cum_sum]
    return first_idx


def map_to_lie_algebra(v):
    """Map a point in R^N to the tangent space at the identity, i.e.
    to the Lie Algebra
    Arg:
        v = vector in R^N, (..., 3) in our case
    Return:
        R = v converted to Lie Algebra element, (3,3) in our case"""

    # make sure this is a sample from R^3
    assert v.size()[-1] == 3

    R_x = v.new_tensor([[0.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])

    R_y = v.new_tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])

    R_z = v.new_tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])

    R = (R_x * v[..., 0, None, None] + R_y * v[..., 1, None, None] + R_z * v[..., 2, None, None])
    return R


def expmap(v):
    theta = v.norm(p=2, dim=-1, keepdim=True)
    # normalize K
    K = map_to_lie_algebra(v / theta)

    I = torch.eye(3, device=v.device, dtype=v.dtype)  # noqa: E741
    R = (I + torch.sin(theta)[..., None] * K + (1.0 - torch.cos(theta))[..., None] * (K @ K))
    return R


# --------
# deal with cryosparc (.cs)
# https://guide.cryosparc.com/setup-configuration-and-management/software-system-guides/
# manipulating-.cs-files-created-by-cryosparc
def cryosparc_to_relion(cs_path, src_star_path, dst_star_path, job_type="homo"):
    if job_type == "abinit":
        rot_key = "alignments_class_0/pose"
        trans_key = "alignments_class_0/shift"
        psize_key = "alignments_class_0/psize_A"
    elif job_type == "homo" or job_type == "hetero":
        rot_key = "alignments3D/pose"
        trans_key = "alignments3D/shift"
        psize_key = "alignments3D/psize_A"
    else:
        raise NotImplementedError(f"Support cryosparc results from abinit (ab-initio reconstruction), "
                                  f"(homo) homogeneous refinement or (hetero) heterogeneous refinements")

    data = np.load(str(cs_path))

    df = starfile.read(src_star_path)

    # view the first row
    for i in range(len(data.dtype)):
        print(i, data.dtype.names[i], data[0][i])

    # parse rotations
    print(f"Extracting rotations from {rot_key}")
    rot = data[rot_key]
    # .copy to avoid negative strides error
    rot = torch.from_numpy(rot.copy())
    rot = expmap(rot)
    rot = rot.numpy()
    print("Transposing rotation matrix")
    rot = rot.transpose((0, 2, 1))
    print(rot.shape)

    #
    # parse translations
    print(f"Extracting translations from {trans_key}")
    trans = data[trans_key]
    if job_type == "hetero":
        print("Scaling shifts by 2x")
        trans *= 2
    pixel_size = data[psize_key]
    # translation in angstroms
    trans *= pixel_size[..., None]
    print(trans.shape)

    # convert to relion
    change_df = df["particles"] if "particles" in df else df
    euler_angles_deg = np.degrees(matrix_to_euler_angles(torch.from_numpy(rot), 'ZYZ').float().numpy())
    change_df["rlnAngleRot"] = -euler_angles_deg[:, 2]
    change_df["rlnAngleTilt"] = -euler_angles_deg[:, 1]
    change_df["rlnAnglePsi"] = -euler_angles_deg[:, 0]

    change_df["rlnOriginXAngst"] = trans[:, 0]
    change_df["rlnOriginYAngst"] = trans[:, 1]

    starfile.write(df, Path(dst_star_path), overwrite=True)


def fft2_center(img):
    pp = np if isinstance(img, np.ndarray) else cp

    return pp.fft.fftshift(pp.fft.fft2(pp.fft.fftshift(img, axes=(-1, -2))), axes=(-1, -2))


def fftn_center(img):
    pp = np if isinstance(img, np.ndarray) else cp

    return pp.fft.fftshift(pp.fft.fftn(pp.fft.fftshift(img)))


def ifftn_center(V):
    pp = np if isinstance(V, np.ndarray) else cp

    V = pp.fft.ifftshift(V)
    V = pp.fft.ifftn(V)
    V = pp.fft.ifftshift(V)
    return V


def ht2_center(img):
    f = fft2_center(img)
    return f.real - f.imag


def htn_center(img):
    pp = np if isinstance(img, np.ndarray) else cp

    f = pp.fft.fftshift(pp.fft.fftn(pp.fft.fftshift(img)))
    return f.real - f.imag


def iht2_center(img):
    img = fft2_center(img)
    img /= img.shape[-1] * img.shape[-2]
    return img.real - img.imag


def ihtn_center(V):
    pp = np if isinstance(V, np.ndarray) else cp

    V = pp.fft.fftshift(V)
    V = pp.fft.fftn(V)
    V = pp.fft.fftshift(V)
    V /= pp.product(V.shape)
    return V.real - V.imag


def _get_start_stop(src_shape, dst_shape):
    start = np.asarray(np.asarray(src_shape) / 2 - np.asarray(dst_shape) / 2, dtype=int)
    stop = start + np.asarray(dst_shape, dtype=int)
    return start, stop


def downsample_vol(vol, dst_shape):
    start, stop = _get_start_stop(vol.shape, dst_shape)

    src_ft = htn_center(vol)
    dst_ft = src_ft[start[0]:stop[0], start[1]:stop[1], start[2]:stop[2]]
    dst_ft = ihtn_center(dst_ft).astype(np.float32)
    return dst_ft


def downsample_images(images, dst_shape):
    src_shape = images.shape[-2:]
    start, stop = _get_start_stop(src_shape, dst_shape)

    src_ft = ht2_center(images)
    dst_ft = iht2_center(src_ft[:, start[0]:stop[0], start[1]:stop[1]]).astype(np.float32)
    return dst_ft


# --------
# visualization
def save_bonds_to_pdb(save_path, atom_arr, bond_pair_ids):
    s = bond_pair_ids.max() + 1
    # remove duplicates
    mask = np.zeros((s, s), dtype=bool)
    np.put(mask, np.ravel_multi_index(bond_pair_ids.T, mask.shape), True)
    mask = np.triu(mask, k=1)
    pair_ids = np.column_stack(np.nonzero(mask))
    tmp_atom_arr = atom_arr.copy()
    tmp_atom_arr.bonds = struc.BondList(len(atom_arr), pair_ids)
    bt_save_pdb(save_path, tmp_atom_arr)


class DeformerProtocol(ABC):

    @abstractmethod
    def transform(self, deformation, coords):
        """
        Input:
            deformation: (bsz, _)
            coords: (num_coords, 3)

        Returns:
            (bsz, num_coords, 3)
        """


class E3Deformer(torch.nn.Module, DeformerProtocol):

    def transform(self, deformation, coords):
        ASSERT_SHAPE(coords, (None, 3))
        ASSERT_SHAPE(deformation, (None, coords.shape[0] * 3))

        bsz = deformation.shape[0]
        shift = deformation.reshape(bsz, -1, 3)
        return shift + coords


class NMADeformer(torch.nn.Module, DeformerProtocol):
    def __init__(self, modes: torch.FloatTensor) -> None:
        super().__init__()
        modes = einops.rearrange(
            modes, "(num_coords c3) num_modes -> num_modes num_coords c3", c3=3
        )
        self.register_buffer("modes", modes)
        self.num_modes = modes.shape[0]
        self.num_coords = modes.shape[1]

    def transform(self, deformation, coords):
        ASSERT_SHAPE(coords, (self.num_coords, 3))
        ASSERT_SHAPE(deformation, (None, 6 + self.num_modes))

        axis_angle = deformation[..., :3]
        translation = deformation[..., 3:6] * 10
        nma_coeff = deformation[..., 6:]
        rotation_matrix = axis_angle_to_matrix(axis_angle)

        nma_deform_e3 = einops.einsum(
            nma_coeff, self.modes, "bsz num_modes, num_modes num_coords c3 -> bsz num_coords c3"
        )
        rotated_coords = einops.einsum(rotation_matrix, nma_deform_e3 + coords,
                                       "bsz c31 c32, bsz num_coords c31 -> bsz num_coords c32")
        deformed_coords = rotated_coords + einops.rearrange(translation, "bsz c3 -> bsz 1 c3")
        return deformed_coords


class VAE(nn.Module):

    def __init__(
        self,
        encoder_cls: str,
        decoder_cls: str,
        in_dim: int,
        e_hidden_dim: Union[int, list, tuple],
        latent_dim: int,
        d_hidden_dim: Union[int, list, tuple],
        out_dim: int,
        e_hidden_layers: int,
        d_hidden_layers: int,
        num_particles: int = -1,
        aa_types: str = None,
    ):
        super().__init__()
        if encoder_cls == "MLP":
            self.encoder = VAEEncoder(in_dim, e_hidden_dim, latent_dim, e_hidden_layers)
        else:
            raise Exception()

        if decoder_cls == "MLP":
            self.decoder = Decoder(latent_dim, d_hidden_dim, out_dim, d_hidden_layers)
        else:
            print(f"{decoder_cls} not in presets, you may set it manually later.")
            self.decoder: torch.nn.Module

    def encode(self, x, idx):
        mean, log_var = self.encoder(x)
        return mean, log_var

    def forward(self, x, idx, *args):
        mean, log_var = self.encode(x, idx)
        z = reparameterize(mean, log_var)
        out = self.decoder(z)
        return out, mean, log_var
