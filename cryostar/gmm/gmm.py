import dataclasses
from typing import Tuple, Union

import einops
import numpy as np
import torch
from torch import nn


@dataclasses.dataclass
class Gaussian:
    mus: Union[torch.Tensor, np.ndarray]
    sigmas: Union[torch.Tensor, np.ndarray]
    amplitudes: Union[torch.Tensor, np.ndarray]


@dataclasses.dataclass
class Grid:
    coords: torch.Tensor  # (N, 1 or 2 or 3)
    shape: Tuple  # (side_shape, ) * 1 or 2 or 3


class EMANGrid(nn.Module):
    """An EMAN style grid which set the origin as -(side_shape // 2)

    """

    def __init__(self, side_shape, voxel_size):
        super().__init__()
        self.side_shape = side_shape

        # integer indices -> angstrom coordinates
        start = -side_shape // 2
        end = side_shape - 1 - side_shape // 2
        line_coords = torch.linspace(start * voxel_size, end * voxel_size, side_shape)
        self.register_buffer("line_coords", line_coords)

        [xx, yy] = torch.meshgrid([line_coords, line_coords], indexing="ij")
        plane_coords = torch.stack([xx, yy], dim=-1).reshape(-1, 2)
        self.register_buffer("plane_coords", plane_coords)
        self.plane_shape = (side_shape, ) * 2

        [xx, yy, zz] = torch.meshgrid([line_coords, line_coords, line_coords], indexing="ij")
        vol_coords = torch.stack([xx, yy, zz], dim=-1).reshape(-1, 3)
        self.register_buffer("vol_coords", vol_coords)
        self.vol_shape = (side_shape, ) * 3

    def line(self):
        return Grid(coords=self.line_coords, shape=(self.side_shape, ))

    def plane(self):
        return Grid(coords=self.plane_coords, shape=self.plane_shape)

    def vol(self):
        return Grid(coords=self.vol_coords, shape=self.vol_shape)


class BaseGrid(nn.Module):
    """Base grid.
    Range from origin (in Angstrom, default (0, 0, 0)), to origin + (side_shape - 1) * voxel_size, almost all data from
    RCSB or EMD follow this convention

    """

    def __init__(self, side_shape, voxel_size, origin=None):
        super().__init__()
        self.side_shape = side_shape
        self.voxel_size = voxel_size

        if origin is None:
            origin = 0
        self.origin = origin

        # integer indices -> angstrom coordinates
        line_coords = torch.linspace(origin, (side_shape - 1) * voxel_size + origin, side_shape)
        self.register_buffer("line_coords", line_coords)

        [xx, yy] = torch.meshgrid([line_coords, line_coords], indexing="ij")
        plane_coords = torch.stack([xx, yy], dim=-1).reshape(-1, 2)
        self.register_buffer("plane_coords", plane_coords)
        self.plane_shape = (side_shape, ) * 2

        [xx, yy, zz] = torch.meshgrid([line_coords, line_coords, line_coords], indexing="ij")
        vol_coords = torch.stack([xx, yy, zz], dim=-1).reshape(-1, 3)
        self.register_buffer("vol_coords", vol_coords)
        self.vol_shape = (side_shape, ) * 3

    def line(self):
        return Grid(coords=self.line_coords, shape=(self.side_shape, ))

    def plane(self):
        return Grid(coords=self.plane_coords, shape=self.plane_shape)

    def vol(self):
        return Grid(coords=self.vol_coords, shape=self.vol_shape)


class NormalGrid(BaseGrid):
    """Normal grid, start from (0,0,0)

    """

    def __init__(self, side_shape, voxel_size):
        super().__init__(side_shape=side_shape, voxel_size=voxel_size, origin=0)


class EMAN2Grid(BaseGrid):
    """EMAN2 style grid.
    origin set to -(side_shape // 2) * voxel_size

    """

    def __init__(self, side_shape, voxel_size):
        origin = -side_shape // 2 * voxel_size
        super().__init__(side_shape=side_shape, voxel_size=voxel_size, origin=origin)


def _expand_batch_dim(t, condition=2):
    if t.ndim == condition:
        t = t[None]
    return t


# For code simplicity, following functions' input args must have a batch dim, notation:
# b: batch_size; nc: num_centers; np: num_pixels; nx, ny, nz: side_shape x, y, z
# gaussian rot with rot_mat then projection along z axis to plane defined by plane or line(x, y)
def batch_projection(gauss: Gaussian, rot_mats: torch.Tensor, line_grid: Grid) -> torch.Tensor:
    """A quick version of e2gmm projection.

    Parameters
    ----------
    gauss: (b/1, num_centers, 3) mus, (b/1, num_centers) sigmas and amplitudes
    rot_mats: (b, 3, 3)
    line_grid: (num_pixels, 3) coords, (nx, ) shape

    Returns
    -------
    proj: (b, y, x) projections
    """

    centers = einops.einsum(rot_mats, gauss.mus, "b c31 c32, b nc c32 -> b nc c31")

    sigmas = einops.rearrange(gauss.sigmas, 'b nc -> b 1 nc')
    sigmas = 2 * sigmas**2

    proj_x = einops.rearrange(line_grid.coords, "nx -> 1 nx 1") - einops.rearrange(centers[..., 0], "b nc -> b 1 nc")
    proj_x = torch.exp(-proj_x**2 / sigmas)

    proj_y = einops.rearrange(line_grid.coords, "ny -> 1 ny 1") - einops.rearrange(centers[..., 1], "b nc -> b 1 nc")
    proj_y = torch.exp(-proj_y**2 / sigmas)

    proj = einops.einsum(gauss.amplitudes, proj_x, proj_y, "b nc, b nx nc, b ny nc -> b nx ny")
    proj = einops.rearrange(proj, "b nx ny -> b ny nx")
    return proj


def batch_canonical_density(gauss: Gaussian, line_grid: Grid):
    sigmas = einops.rearrange(gauss.sigmas, "b nc -> b 1 nc 1")
    sigmas = 2 * sigmas**2

    proj_xyz = einops.rearrange(line_grid.coords, "nx -> 1 nx 1 1") - \
               einops.rearrange(gauss.mus, 'b nc c3 -> b 1 nc c3')
    proj_xyz = torch.exp(-proj_xyz**2 / sigmas)

    vol = einops.einsum(gauss.amplitudes, proj_xyz[..., 0], proj_xyz[..., 1], proj_xyz[..., 2],
                        "b nc, b nx nc, b ny nc, b nz nc -> b nx ny nz")
    return vol


def canonical_density(gauss: Gaussian, line_grid: Grid):
    vol = batch_canonical_density(
        Gaussian(mus=gauss.mus[None], sigmas=gauss.sigmas[None], amplitudes=gauss.amplitudes[None]), line_grid)
    return vol.squeeze(0)


def batch_density(gauss: Gaussian, vol_grid: Grid):
    xyz = einops.rearrange(vol_grid.coords, 'np c3 -> 1 np 1 c3')  # (b, np, nc, 3)
    centers = einops.rearrange(gauss.mus, 'b nc c3 -> b 1 nc c3')
    sigmas = einops.rearrange(gauss.sigmas, 'b nc -> b 1 nc')
    amplitudes = einops.rearrange(gauss.amplitudes, 'b nc -> b 1 nc')

    delta = xyz - centers
    p = amplitudes * torch.exp(-(delta**2).sum(-1) / (2 * sigmas**2))  # (b, np, nc)
    p = p.sum(-1)
    p = p.reshape(vol_grid.shape)
    return p


# for backup, need modification
def batch_ft_projection(gauss: Gaussian, rot_mats: torch.Tensor, freqs: torch.Tensor):
    centers = einops.einsum(rot_mats, gauss.mus, "b c31 c32, b nc c32 -> b nc c31")

    sigmas = einops.rearrange(gauss.sigmas, 'b nc -> b nc 1')

    freqs = einops.rearrange(freqs, "k -> 1 1 k")

    # canonical gaussian fourier transformation, (b, nc, k)
    f_proj_base = math.sqrt(2 * torch.pi) * sigmas * torch.exp(-2 * torch.pow(torch.pi * sigmas * freqs, 2))

    # shift by center x, y
    f_proj_x = f_proj_base * torch.exp(-1j * 2 * torch.pi * centers[..., 0:1] * freqs)
    f_proj_y = f_proj_base * torch.exp(-1j * 2 * torch.pi * centers[..., 1:2] * freqs)

    f_proj = einops.einsum(gauss.amplitudes, f_proj_x, f_proj_y, "b nc, b nc kx, b nc ky -> b ky kx")
    # you may need another normalization term sqrt(2 * side_shape) to match other DFT results
    return f_proj
