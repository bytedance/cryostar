from typing import Tuple, Union
import mrcfile
import numpy as np

import torch
from torchvision.transforms import ToPILImage
from torchvision.utils import make_grid


def save_mrc(vol,
             path,
             voxel_size: Union[int, float, Tuple, np.recarray] = None,
             origin: Union[int, float, Tuple, np.recarray] = None):
    """
    Save volumetric data to mrc file, set voxel_size, origin.
    See Also: https://mrcfile.readthedocs.io/en/stable/source/mrcfile.html#mrcfile.mrcobject.MrcObject.voxel_size
    Args:
        vol: density volume
        path: save path
        voxel_size: a single number, a 3-tuple (x, y ,z) or a modified version of the voxel_size array, default 1.
        origin: a single number, a 3-tuple (x, y ,z) or a modified version of the origin array, default 0.

    """
    with mrcfile.new(path, overwrite=True) as m:
        m.set_data(vol)

        if voxel_size is not None:
            m.voxel_size = voxel_size

        if origin is not None:
            m.header.origin = origin


def show_particles(file_path, num=25, nrow=5, permissive=False):
    with mrcfile.mmap(file_path, permissive=permissive) as m:
        vol = m.data
        total_particles = vol.shape[0]
        print(f"Total {total_particles} particles")

    total_num = min(total_particles, num)

    show_img = ToPILImage()(make_grid(torch.from_numpy(vol[:total_num].copy()).unsqueeze(1), nrow=nrow, normalize=True))
    return show_img
