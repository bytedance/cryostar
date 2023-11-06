import einops
import numpy as np
from scipy import interpolate, ndimage
from scipy.optimize import linear_sum_assignment

import biotite.structure as struc

from .misc import convert_to_numpy


def get_Ca_RMSE(cas_ref, cas_hat, assignment="order"):
    """
        Input:
            assignment: how to assign two models?
                - order:    supposing the i-th vector in the reference is corresponding
                                to the i-th one in the prediction
                - recall:   we only care about how many vectors are recalled
                - align:    we align the reference with the prediction by mean-squared-error
    """
    assert assignment in ("order", "recall", "align")
    cas_ref, cas_hat = convert_to_numpy(cas_ref, cas_hat)
    if assignment == "order":
        assert len(cas_ref) == len(cas_hat), "Ordered RMSE requires cas1 and cas2 to have the same shape."
        delta = cas_ref - cas_hat
        rmses = einops.reduce(delta**2, "centers 3 -> centers", reduction="sum")**0.5
        rmse = rmses.mean()
        within3 = (rmses < 3).sum() / len(rmses)
        return {"rmse-order": rmse, "within3-order": within3}
    else:
        cas1_square = einops.reduce(cas_ref**2, "B T -> B ()", "sum")
        cas2_square = einops.reduce(cas_hat**2, "B T -> () B", "sum")
        cas12_inner = einops.einsum(cas_ref, cas_hat, "B1 D, B2 D -> B1 B2")
        delta = (cas1_square + cas2_square - 2 * cas12_inner + 1e-3)**0.5
        assert (np.linalg.norm(cas_ref[0, :] - cas_hat, axis=1) - delta[0]).sum() < 1e-2
        if assignment == "recall":
            delta_min = einops.reduce(delta, "B1 B2 -> B1", "min")
            rmse = delta_min.mean()
            within3 = (delta_min < 3).sum() / len(delta)
            return {"rmse-recall": rmse, "within3-recall": within3}
        elif assignment == "align":
            row_ind, col_ind = linear_sum_assignment(delta)
            delta_min = delta[row_ind, col_ind]
            rmse = delta_min.mean()
            within3 = (delta_min < 3).sum() / len(delta)
            return {"rmse-align": rmse, "within3-align": within3}


def get_vol_NMSE(vol_gold, vol_pred):
    flat_gold = vol_gold.flatten()
    flat_pred = vol_pred.flatten()
    nmse = ((flat_pred - flat_gold)**2).sum() / (flat_gold**2).sum()
    return {"nmse": nmse}


def get_FSC(vol1, vol2):
    vol1, vol2 = convert_to_numpy(vol1, vol2)
    # 原来的 code base 里，代码可能有 bug
    vol1_ft = np.fft.fftshift(np.fft.fftn(np.fft.fftshift(vol1)))
    vol2_ft = np.fft.fftshift(np.fft.fftn(np.fft.fftshift(vol2)))
    # vol1_ft = np.fft.fftshift(np.fft.fftn(vol1))
    # vol2_ft = np.fft.fftshift(np.fft.fftn(vol2))

    # 对于 DxDxD 的 Fourier Volume，将中心放在原点，半径是 D//2
    # 按照球壳半径（整数）的大小对 Volume 的每个位置打标签（半径）
    D = vol1.shape[0]
    x = np.arange(-D // 2, D // 2)
    x0, x1, x2 = np.meshgrid(x, x, x, indexing='ij')
    r = np.sqrt(x0**2 + x1**2 + x2**2)
    r_max = D // 2  # sphere inscribed within volume box
    r_step = 1  # int(np.min(r[r>0]))
    bins = np.arange(0, r_max, r_step)
    bin_labels = np.searchsorted(bins, r, side='right')

    # 对于每个（半径）标签下的区域计算 FSC
    # 超过半径的不予计算
    num = ndimage.sum(np.real(vol1_ft * np.conjugate(vol2_ft)), labels=bin_labels, index=bins + 1)
    den1 = ndimage.sum(np.abs(vol1_ft)**2, labels=bin_labels, index=bins + 1)
    den2 = ndimage.sum(np.abs(vol2_ft)**2, labels=bin_labels, index=bins + 1)
    fsc = num / np.sqrt(den1 * den2)
    # x axis should be spatial frequency in 1/px
    x = bins / D

    # 拟合一个 fsc 曲线，解出第一个 0.143
    fsc_func = interpolate.interp1d(x, fsc)
    x_new = np.arange(min(x), max(x), 0.001)
    y_new = fsc_func(x_new)
    delta = np.abs(y_new - 0.143)
    if all(delta >= 0.01):
        selected_idx = np.argmin(delta)
    else:
        selected_idx = np.where(delta < 0.01)[0][0]
    resolution = 1 / x_new[selected_idx]

    return {"x": x_new, "y": y_new, "func": fsc_func, "resolution": resolution}


def calc_coverage(ref_set, gen_set, delta=0.5, do_align=True):
    num_ref = len(ref_set)
    # num_gen = len(gen_set)

    count = 0

    for i in range(num_ref):
        tmp = gen_set
        if do_align:
            tmp, _ = struc.superimpose(ref_set[i], tmp)
        if np.any(struc.rmsd(ref_set[i], tmp) < delta):
            count += 1
    return count / num_ref


def calc_mat(ref_set, gen_set, do_align=True):
    num_ref = len(ref_set)
    # num_gen = len(gen_set)

    numerator = 0.

    for i in range(num_ref):
        tmp = gen_set
        if do_align:
            tmp, _ = struc.superimpose(ref_set[i], tmp)
        numerator += np.amin(struc.rmsd(ref_set[i], tmp))
    return numerator / num_ref
