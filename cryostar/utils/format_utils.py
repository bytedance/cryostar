from pathlib import Path

import numpy as np
import starfile
import torch

from cryostar.utils.rotation_conversion import matrix_to_euler_angles


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