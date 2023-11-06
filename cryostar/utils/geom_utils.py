import numpy as np
import torch

from cryostar.utils.rotation_conversion import euler_angles_to_matrix, matrix_to_euler_angles


def euler_angles2matrix(alpha, beta, gamma):
    """
    Converts euler angles in RELION convention to rotation matrix.

    Parameters
    ----------
    alpha: float / np.array
    beta: float / np.array
    gamma: float / np.array

    Returns
    -------
    A: np.array (3, 3)
    """
    # For RELION Euler angle convention
    ca = np.cos(alpha)
    cb = np.cos(beta)
    cg = np.cos(gamma)
    sa = np.sin(alpha)
    sb = np.sin(beta)
    sg = np.sin(gamma)
    cc = cb * ca
    cs = cb * sa
    sc = sb * ca
    ss = sb * sa

    A = np.zeros((3, 3))
    A[0, 0] = cg * cc - sg * sa
    A[0, 1] = -cg * cs - sg * ca
    A[0, 2] = cg * sb
    A[1, 0] = sg * cc + cg * sa
    A[1, 1] = -sg * cs + cg * ca
    A[1, 2] = sg * sb
    A[2, 0] = -sc
    A[2, 1] = ss
    A[2, 2] = cb
    return A


def get_rotation_accuracy(rot_gt, rot_pred, N=30):
    """
    Computes the MSE and MedSE between gt and predicted (aligned) rotations.

    Parameters
    ----------
    rot_gt: np.array (N, 3, 3)
    rot_pred: np.array (N, 3, 3)
    N: int

    Returns
    -------
    MSE: float
    MedSE: float
    """
    mean = []
    median = []
    for i in range(N):
        rot_pred_aligned = align_rot(rot_gt, rot_pred, i, flip=False)
        dists = np.sum((rot_gt - rot_pred_aligned)**2, axis=(1, 2))
        mean.append(np.mean(dists))
        median.append(np.median(dists))

    mean_flip = []
    median_flip = []
    for i in range(N):
        rot_pred_aligned = align_rot(rot_gt, rot_pred, i, flip=True)
        dists = np.sum((rot_gt - rot_pred_aligned)**2, axis=(1, 2))
        mean_flip.append(np.mean(dists))
        median_flip.append(np.median(dists))

    MSE = np.min(np.concatenate([mean, mean_flip]))
    MedSE = np.min(np.concatenate([median, median_flip]))

    return MSE, MedSE


def get_ref_matrix(r1, r2, i, flip=False):
    """
    Returns the rotation R such that r2[i].R = r1[i].

    Parameters
    ----------
    r1: np.array (N, 3, 3)
    r2: np.array (N, 3, 3)
    i: int
    flip: bool

    Returns
    -------
    R: np.array(3, 3)
    """
    if flip:
        return np.matmul(r2[i].T, _flip(r1[i]))
    else:
        return np.matmul(r2[i].T, r1[i])


def _flip(rot):
    """
    Flips the rotation rot.

    Parameters
    ----------
    rot: np.array (3, 3)

    Returns
    -------
    out: np.array (3, 3)
    """
    x = np.diag([1, 1, -1]).astype(rot.dtype)
    return np.matmul(x, rot)


def align_rot(r1, r2, i, flip=False):
    """
    Aligns r2 on r1 with index i.

    Parameters
    ----------
    r1: np.array (N, 3, 3)
    r2: np.array (N, 3, 3)
    i: int
    flip: bool

    Returns
    -------
    out: np.array (N, 3, 3)
    """
    if flip:
        return np.matmul(_flip(r2), get_ref_matrix(r1, r2, i, flip=True))
    else:
        return np.matmul(r2, get_ref_matrix(r1, r2, i, flip=False))


def correct_flips_rotations(model_output):
    """
    Corrects the in-plane flip due to the symmetrized loss.

    Parameters
    ----------
    model_output: Dictionary
    batch_size: int

    Returns
    -------
    rotmat_pred: torch.Tensor (B, 3, 3)
    """
    rotmat_pred = model_output['rotmat'].reshape(2, -1, 3, 3)  # 2, B, 3, 3
    batch_size = rotmat_pred.shape[1]
    idx = model_output['activated_paths'].reshape(1, batch_size, 1, 1).repeat(1, 1, 3, 3)
    rotmat_pred = rotmat_pred.gather(0, idx).reshape(batch_size, 3, 3)  # B, 3, 3
    euler_pred = matrix_to_euler_angles(rotmat_pred, 'ZYZ')  # B, 3
    additional_alpha = torch.tensor([np.pi] * batch_size).cuda() * model_output['activated_paths']  # N
    euler_pred[:, 0] = euler_pred[:, 0] + additional_alpha
    rotmat_pred = euler_angles_to_matrix(euler_pred, 'ZYZ')  # B, 3, 3
    return rotmat_pred


def add_flipped_rotmat(pred_rotmat):
    """
    Concatenates batch wise ground truth rotation matrices and their flipped version.

    Parameters
    ----------
    pred_rotmat: torch.Tensor (B, 3, 3)

    Returns
    -------
    out: torch.Tensor (2*B, 3, 3)
    """
    euler_angles = matrix_to_euler_angles(pred_rotmat, 'ZYZ')  # B, 3
    additional_pi = torch.tensor([np.pi, 0., 0.]).reshape(1, 3).to(euler_angles.device)
    euler_angles = euler_angles + additional_pi
    pred_rotmat_flipped = euler_angles_to_matrix(euler_angles, 'ZYZ')
    return torch.cat([pred_rotmat, pred_rotmat_flipped], dim=0)
