from functools import cmp_to_key

import numpy as np
from scipy.spatial import distance

import torch
from torch import nn
import torch.linalg as LA

from cryostar.utils.polymer import AA_ATOMS, NT_ATOMS


def calc_dist_by_pair_indices(coord_arr, pair_indices):
    coord_pair_arr = coord_arr[pair_indices]  # num_pair, 2, 3
    dist = np.linalg.norm(np.diff(coord_pair_arr, axis=1), ord=2, axis=-1)
    return dist.flatten()


def find_continuous_pairs(chain_id_arr, res_id_arr, atom_name_arr):
    pairs = []

    # res_id in different chains are duplicated, so loop on chains
    u_chain_id = np.unique(chain_id_arr)

    for c_id in u_chain_id:
        tmp_mask = chain_id_arr == c_id
        tmp_indices_in_pdb = np.nonzero(tmp_mask)[0]

        tmp_res_id_arr = res_id_arr[tmp_mask]
        tmp_atom_name_arr = atom_name_arr[tmp_mask]

        # check is aa or nt
        tmp_atom_name_set = set(tmp_atom_name_arr)

        if len(tmp_atom_name_set.intersection(AA_ATOMS)) > len(tmp_atom_name_set.intersection(NT_ATOMS)):
            in_res_atom_names = AA_ATOMS
        elif len(tmp_atom_name_set.intersection(AA_ATOMS)) < len(tmp_atom_name_set.intersection(NT_ATOMS)):
            in_res_atom_names = NT_ATOMS
        else:
            raise NotImplemented("Cannot determine chain is amino acid or nucleotide.")

        # find pairs
        if len(in_res_atom_names) == 1:
            u_res_id, indices_in_chain = np.unique(tmp_res_id_arr, return_index=True)
            if len(u_res_id) != np.sum(tmp_mask):
                raise ValueError(f"Found duplicate residue id in single chain {c_id}.")

            indices_in_chain_pair = np.column_stack((indices_in_chain[:-1], indices_in_chain[1:]))

            # must be adjacent on residue id
            valid_mask = np.abs(np.diff(u_res_id[indices_in_chain_pair], axis=1)) == 1

            indices_in_chain_pair = indices_in_chain_pair[valid_mask.flatten()]

            indices_in_pdb_pair = tmp_indices_in_pdb[indices_in_chain_pair]
        elif len(in_res_atom_names) > 1:

            def _cmp(a, b):
                # res_id compare
                if a[0] != b[0]:
                    return a[0] - b[0]
                else:
                    # atom_name in the same order of AA_ATOMS or NT_ATOMS
                    return in_res_atom_names.index(a[1]) - in_res_atom_names.index(b[1])

            cache = list(zip(tmp_res_id_arr, tmp_atom_name_arr, tmp_indices_in_pdb))
            sorted_cache = list(sorted(cache, key=cmp_to_key(_cmp)))

            sorted_indices_in_pdb = [item[2] for item in sorted_cache]
            sorted_res_id = [item[0] for item in sorted_cache]

            indices_in_pdb_pair = np.column_stack((sorted_indices_in_pdb[:-1], sorted_indices_in_pdb[1:]))

            valid_mask = np.abs(np.diff(np.column_stack((sorted_res_id[:-1], sorted_res_id[1:])), axis=1)) <= 1

            indices_in_pdb_pair = indices_in_pdb_pair[valid_mask.flatten()]
        else:
            raise NotImplemented("No enough atoms to construct continuous pairs.")

        pairs.append(indices_in_pdb_pair)

    pairs = np.vstack(pairs)
    return pairs


def find_quaint_cutoff_pairs(coord_arr,
                             chain_id_arr,
                             res_id_arr,
                             intra_chain_cutoff=12.,
                             inter_chain_cutoff=12.,
                             intra_chain_res_bound=None):
    sel_indices = []
    dist_map = distance.cdist(coord_arr, coord_arr, metric='euclidean')
    # 1. intra chain
    sel_mask = dist_map <= intra_chain_cutoff
    sel_mask = np.triu(sel_mask, k=1)
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
        sel_mask = np.triu(sel_mask, k=1)
        indices_in_pdb = np.nonzero(sel_mask)
        indices_in_pdb = np.column_stack((indices_in_pdb[0], indices_in_pdb[1]))
        indices_in_pdb = indices_in_pdb[chain_id_arr[indices_in_pdb[:, 0]] != chain_id_arr[indices_in_pdb[:, 1]]]
        sel_indices.append(indices_in_pdb)

    sel_indices = np.vstack(sel_indices)
    return sel_indices


def find_range_cutoff_pairs(coord_arr, min_cutoff=4., max_cutoff=10.):
    dist_map = distance.cdist(coord_arr, coord_arr, metric='euclidean')
    sel_mask = (dist_map <= max_cutoff) & (dist_map >= min_cutoff)
    indices_in_pdb = np.nonzero(sel_mask)
    indices_in_pdb = np.column_stack((indices_in_pdb[0], indices_in_pdb[1]))
    return indices_in_pdb


def remove_duplicate_pairs(pairs_a, pairs_b, remove_flip=True):
    """Remove pair b from a"""
    s = max(pairs_a.max(), pairs_b.max()) + 1
    # trick for fast comparison
    mask = np.zeros((s, s), dtype=bool)

    np.put(mask, np.ravel_multi_index(pairs_a.T, mask.shape), True)
    np.put(mask, np.ravel_multi_index(pairs_b.T, mask.shape), False)
    if remove_flip:
        np.put(mask, np.ravel_multi_index(np.flip(pairs_b, 1).T, mask.shape), False)
    return np.column_stack(np.nonzero(mask))


def filter_same_chain_pairs(pair_ids, chain_id_arr):
    chain_ids = chain_id_arr[pair_ids]

    same_chain_mask = chain_ids[:, 0] == chain_ids[:, 1]

    pair_mask = []

    for u in np.unique(chain_ids):
        tmp = np.logical_and(chain_ids[:, 0] == u, same_chain_mask)
        if np.any(tmp):
            pair_mask.append(tmp)

    if len(pair_mask) > 0:
        return np.row_stack(pair_mask)
    else:
        return None


class DistLoss(nn.Module):

    def __init__(self, pair_ids, gt_dists, reduction="mean"):
        super().__init__()
        self.reduction = reduction

        self.register_buffer("pair_ids", torch.from_numpy(pair_ids).long())
        self.register_buffer("gt_dists", torch.from_numpy(gt_dists).float())

        # edge-wise weights
        # raw_weights = torch.ones(len(pair_ids), dtype=torch.float) * 3.
        #
        # self.register_parameter("raw_weights", nn.Parameter(raw_weights))

        # RBF residue-wise weights
        # u_left_ids = np.unique(pair_ids[:, 0])
        #
        # std_idx = np.zeros(max(u_left_ids) + 1, dtype=int)
        # sparse_idx = np.arange(len(u_left_ids))
        #
        # std_idx[u_left_ids] = sparse_idx
        #
        # select_index = std_idx[pair_ids[:, 0]]

        # weight = 0.9 at dist_rescale
        # sigmas = torch.ones(max(u_left_ids) + 1, dtype=torch.float) * np.sqrt(-0.5 / np.log(0.9))
        #
        # self.dist_rescale = dist_rescale
        # self.register_buffer("select_index", torch.from_numpy(select_index).long())
        # self.register_parameter("sigmas", nn.Parameter(sigmas))

    # def get_weights(self):
    # return torch.sigmoid(self.raw_weights)
    # edge_sigmas = torch.index_select(self.sigmas, dim=0, index=self.select_index)
    # weights = torch.exp(-torch.pow(self.gt_dists / self.dist_rescale, 2) / (2 * torch.pow(edge_sigmas, 2)))
    # return weights

    def calc_pair_dists(self, batch_struc):
        batch_dist = batch_struc[:, self.pair_ids]  # bsz, num_pair, 2, 3
        batch_dist = LA.vector_norm(torch.diff(batch_dist, dim=-2), axis=-1).squeeze(-1)  # bsz, num_pair
        return batch_dist

    def forward(self, batch_struc):
        batch_dist = self.calc_pair_dists(batch_struc)
        # mse = torch.pow(batch_dist - self.gt_dists.unsqueeze(0), 2) * self.get_weights().unsqueeze(0)
        mse = torch.pow(batch_dist - self.gt_dists.unsqueeze(0), 2)
        if self.reduction is None:
            return mse
        elif self.reduction == "mean":
            return torch.mean(mse)
        else:
            raise NotImplementedError
