import dataclasses
from abc import abstractmethod
from typing import Tuple, Union

import einops
import numpy as np
import torch
from torch import nn

from cryostar.utils.misc import ASSERT_SHAPE


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


class BaseGrid2(nn.Module):

    def __init__(self, start, stop, num, endpoint=True):
        super().__init__()

        line_coords = torch.from_numpy(np.linspace(start, stop, num, endpoint=endpoint)).float()
        self.register_buffer("line_coords", line_coords)
        self.line_shape = (num, )

        [xx, yy] = torch.meshgrid([line_coords, line_coords], indexing="ij")
        plane_coords = torch.stack([xx, yy], dim=-1).reshape(-1, 2)
        self.register_buffer("plane_coords", plane_coords)
        self.plane_shape = (num, ) * 2

        [xx, yy, zz] = torch.meshgrid([line_coords, line_coords, line_coords], indexing="ij")
        vol_coords = torch.stack([xx, yy, zz], dim=-1).reshape(-1, 3)
        self.register_buffer("vol_coords", vol_coords)
        self.vol_shape = (num, ) * 3

    def line(self):
        return Grid(coords=self.line_coords, shape=self.line_shape)

    def plane(self):
        return Grid(coords=self.plane_coords, shape=self.plane_shape)

    def vol(self):
        return Grid(coords=self.vol_coords, shape=self.vol_shape)


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


def batch_e2gmm_projection(gauss: Gaussian, rot_mats: torch.Tensor, plane_grid: Grid) -> torch.Tensor:
    """An e2gmm style projection implementation, backward compatible
    ref: Deep learning-based mixed-dimensional Gaussian mixture model for characterizing variability in cryo-EM

    Parameters
    ----------
    gauss: (b, num_centers, 3) mus, (b, num_centers) sigmas and amplitudes
    rot_mats: (b, 3, 3)
    plane_grid: (num_pixels, 3) coords, (ny, nx) shape

    Returns
    -------
    p: (b, ny, nx) projections
    """
    # the rotation matrix is used to rotate the particle (object) by R * (3 x 1 xyz coordinates),
    # following the e2gmm paper, the third row is dropped, so the equation should be
    # (n x 3 coordinates) * (R(2 x 3)).T, here the einsum will transpose the rotation matrix
    p = torch.einsum('ijk,imk -> ijm', gauss.mus, rot_mats[:, :2])
    p = einops.rearrange(plane_grid.coords, 'np c -> 1 np 1 c') - einops.rearrange(p, 'b nc c -> b 1 nc c')
    p = torch.pow(p, 2)
    # p.pow_(2)
    p = -p.sum(-1)
    p /= 2 * einops.rearrange(gauss.sigmas, 'b nc -> b 1 nc')**2
    # p.div_(2 * einops.rearrange(sigmas, 'b nc -> b 1 nc') ** 2)
    p = torch.exp(p)
    p = p * einops.rearrange(gauss.amplitudes, 'b nc -> b 1 nc')
    # p.mul_(einops.rearrange(amplitudes, 'b nc -> b 1 nc'))

    p = p.sum(-1)
    # transpose x, y -> y, x
    p = einops.rearrange(p, 'b (nx ny) -> b ny nx', nx=plane_grid.shape[1], ny=plane_grid.shape[0])
    return p


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


class AbsGMM(nn.Module):

    @abstractmethod
    def get_centers(self):
        pass

    @abstractmethod
    def get_amplitudes(self):
        pass

    @abstractmethod
    def get_sigmas(self):
        pass


class ProjectorBase(nn.Module):

    def __init__(self, side_shape, voxel_size, device=None):
        """

        Parameters
        ----------
        side_shape (int): control generated particle and volume shape
        voxel_size (float): control the particle pixel size or volume voxel_size in angstrom
        """
        super().__init__()
        self.side_shape = side_shape

        # integer indices -> angstrom coordinates
        start = -side_shape // 2
        end = side_shape - 1 - side_shape // 2
        line_coords = torch.linspace(start * voxel_size, end * voxel_size, side_shape).to(device)
        self.register_buffer("line_coords", line_coords)

        [xx, yy] = torch.meshgrid([line_coords, line_coords], indexing="ij")
        plane_coords = torch.stack([xx, yy], dim=-1).reshape(-1, 2).to(device)
        self.register_buffer("plane_coords", plane_coords)
        self.plane_shape = (side_shape, ) * 2

        [xx, yy, zz] = torch.meshgrid([line_coords, line_coords, line_coords], indexing="ij")
        vol_coords = torch.stack([xx, yy, zz], dim=-1).reshape(-1, 3).to(device)
        self.register_buffer("vol_coords", vol_coords)
        self.vol_shape = (side_shape, ) * 3


class GMMProjector(ProjectorBase):

    def density(self, gmm: AbsGMM, xyz: torch.FloatTensor):
        ASSERT_SHAPE(xyz, (None, 3))

        delta = xyz[:, None, :] - gmm.get_centers()  # (N_points, N_centers, 3)
        p = gmm.get_amplitudes() * torch.exp(-(delta**2).sum(-1) / (2 * gmm.get_sigmas()**2))  # (N_points, N_centers)
        p = p.sum(-1)  # (N_points, )
        return p

    def canonical_density(self, gmm: AbsGMM):
        """
            Evaluate the canonical density of a GMM in XYZ order.

            Returns:
                density:    (NX, NY, NZ)
        """
        centers = gmm.get_centers()

        sigmas = einops.rearrange(gmm.get_sigmas(), "N -> 1 N 1")
        sigmas = 2 * sigmas**2

        proj_xyz = einops.rearrange(self.line_coords, "D -> D 1 1") \
                   - einops.rearrange(centers, "N C3 -> 1 N C3")
        proj_xyz = torch.exp(-proj_xyz**2 / sigmas)

        proj = einops.einsum(proj_xyz[..., 0], proj_xyz[..., 1], proj_xyz[..., 2], gmm.get_amplitudes(),
                             "D1 N, D2 N, D3 N, N -> D1 D2 D3")

        return proj

    def canonical_density_slow(self, gmm: AbsGMM):
        """
            Evaluate the canonical density of a GMM in XYZ order.

            Returns:
                density:    (NX, NY, NZ)
        """
        p = self.density(gmm, self.vol_coords)
        # axis is the same order as vol_coords
        p = p.reshape(self.vol_shape)
        return p

    def project(self, gmm: AbsGMM, rot_mats: torch.FloatTensor):
        """

            Input:
                rot_mats:       (batch_size, 3, 3)

            Returns:
                images:         (batch_size, NY, NX)
        
        """
        ASSERT_SHAPE(rot_mats, (None, 3, 3))

        rot_centers = einops.einsum(rot_mats, gmm.get_centers(), "bsz C31 C32, N C32 -> bsz N C31")

        sigmas = einops.rearrange(gmm.get_sigmas(), "N -> 1 1 N")
        sigmas = 2 * sigmas**2

        proj_x = einops.rearrange(self.line_coords, "D -> 1 D 1") \
                 - einops.rearrange(rot_centers[..., 0], "B N -> B 1 N")
        proj_x = torch.exp(-proj_x**2 / sigmas)

        proj_y = einops.rearrange(self.line_coords, "D -> 1 D 1") \
                 - einops.rearrange(rot_centers[..., 1], "B N -> B 1 N")
        proj_y = torch.exp(-proj_y**2 / sigmas)

        proj = einops.einsum(proj_x, proj_y, gmm.get_amplitudes(), "B D1 N, B D2 N, N -> B D1 D2")
        proj = einops.rearrange(proj, "B NX NY -> B NY NX")
        return proj

    def project_numerical_1(self, gmm: AbsGMM, rot_mats: torch.FloatTensor):
        # rotate the sampling grid then sum along z axis, calc point value analytically
        ASSERT_SHAPE(rot_mats, (None, 3, 3))

        bsz = rot_mats.shape[0]
        # the rotation matrix is defined to rotate the object, so here the grid coordinates should be
        # rotated by R^{-1} = R.T, R.T * (3 x n coordinates) -> (n x 3 coordinates) * (R.T).T = (n x 3 coordinates) * R
        xyz_rotated = einops.einsum(self.vol_coords.unsqueeze(0), rot_mats,
                                    "bsz coords C31, bsz C31 C32 -> bsz coords C32")
        p = self.density(gmm, xyz_rotated.reshape(-1, 3)).reshape((bsz, *self.vol_shape))
        p = p.sum(-1)
        p = torch.transpose(p, -2, -1)
        return p

    def project_numerical_2(self, gmm: AbsGMM, rot_mats: torch.FloatTensor):
        # rotate the sampling grid then sum along z axis, sample point value from canonical density
        ASSERT_SHAPE(rot_mats, (None, 3, 3))
        """
            See: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
            In grid sampling, the shape of `input` is [N, C, D1, H1, W1], the shape of `grid` is
            [N, D2, H2, W2, 3]. Note that grid[n, d, h, w] is a 3-vector (wx, hx, dx) which is
            corresponding to (W, H, D) respecitvely, which means an **identity** map should be:
                grid[d, h, w] = (wx, hx, dx)
                     |     |     |       |
                     |     |     |       slow axis
                     |     |     fast axis
                     |     |
                     |     fast axis
                     slow axis
            If we made some mistake, the axes will be exchanged:
                output[~]:  000 001 002 003 010 011 012 013
                input[~]:   000 100 200 300 010 110 210 310
        """
        bsz = rot_mats.shape[0]

        # p[d, h, w] is a scalar
        d = self.canonical_density(gmm)

        # c[d, h, w] is a 3-vector
        # here the equation (n x 3 coordinates) * R behaves the same as above function, the flip is because
        # pytorch grid_sample index is in reverse order of input tensor
        """
            Checked (no-batch case):
                    |  |  |    -R1-          |
            c @ R = c1 c2 c3 @ -R2- = \sum_i ci @ -Ri- (each item shaped c.row x R.col)
                    |  |  |    -R3-          |
            (c @ R).flip(1) = c @ R.flip(1)
        """
        c = self.vol_coords.unsqueeze(0) / (self.side_shape // 2) @ rot_mats.flip(2)
        c = c.reshape(bsz, *self.vol_shape, 3)

        p = torch.nn.functional.grid_sample(
            d[None, None].expand(bsz, -1, -1, -1, -1),  #
            c,
            align_corners=True,
        )

        p = p.reshape(bsz, *self.vol_shape)
        p = p.sum(-1)
        p = torch.transpose(p, -2, -1)
        return p

    def project_slow(self, gmm: AbsGMM, rot_mats: torch.FloatTensor):
        """

            Input:
                rot_mats:       (batch_size, 3, 3)

            Returns:
                images:         (batch_size, NY, NX)
        
        """
        return batch_e2gmm_projection(gmm.get_centers(), gmm.get_sigmas(), gmm.get_amplitudes(), rot_mats,
                                      self.plane_coords, self.plane_shape)


class ShiftedGMMProjector(ProjectorBase):

    def density(self, gmm: AbsGMM, xyz: torch.FloatTensor, shift: torch.FloatTensor):
        ASSERT_SHAPE(xyz, (None, 3))
        ASSERT_SHAPE(shift, (None, None, 3))

        # batch_size: the number of deformations
        centers = gmm.get_centers().unsqueeze(0) + shift
        delta = xyz[None, :, None, :] - centers[:, None, :, :]  # (batch_size, N_points, N_centers, 3)
        p = gmm.get_amplitudes() * torch.exp(-(delta**2).sum(-1) /
                                             (2 * gmm.get_sigmas()**2))  # (batch_size, N_points, N_centers)
        p = p.sum(-1)  # (batch_size, N_points, )
        return p

    def canonical_density(self, gmm: AbsGMM, shift: torch.FloatTensor):
        """
            Evaluate a batch of canonical densities of a GMM in XYZ order.

            Input:
                shift:      (B, T, 3)
            
            Returns:
                density:    (B, NX, NY, NZ)
        """
        B, T, _ = shift.shape

        centers = einops.repeat(gmm.get_centers(), "N C3 -> B N C3", B=B) + shift

        sigmas = einops.rearrange(gmm.get_sigmas(), "N -> 1 1 N 1")
        sigmas = 2 * sigmas**2

        proj_xyz = einops.rearrange(self.line_coords, "D -> 1 D 1 1") \
                   - einops.rearrange(centers, "B N C3 -> B 1 N C3")
        proj_xyz = torch.exp(-proj_xyz**2 / sigmas)

        proj = einops.einsum(proj_xyz[..., 0], proj_xyz[..., 1], proj_xyz[..., 2], gmm.get_amplitudes(),
                             "B D1 N, B D2 N, B D3 N, N -> B D1 D2 D3")

        return proj

    def canonical_density_slow(self, gmm: AbsGMM, shift: torch.FloatTensor):
        """
            Evaluate a batch of canonical densities of a GMM in XYZ order.

            Input:
                shift:      (B, T, 3)
            
            Returns:
                density:    (B, NX, NY, NZ)
        """
        ASSERT_SHAPE(shift, (None, None, 3))

        p = self.density(gmm, self.vol_coords, shift)
        # axis is the same order as vol_coords
        p = p.reshape((p.shape[0], ) + self.vol_shape)
        return p

    def project(self, gmm: AbsGMM, rot_mats: torch.FloatTensor, shift: torch.FloatTensor):
        """
            `rot_mats[i]` is binded with `shift[i]`.

            Inputs:
                rot_mats:       (B, 3, 3)
                shift:          (B, T, 3)

            Returns:
                (B, NY, NX)
        """
        ASSERT_SHAPE(rot_mats, (None, 3, 3))
        ASSERT_SHAPE(shift, (None, None, 3))
        assert shift.shape[0] == rot_mats.shape[0]
        B = rot_mats.shape[0]

        centers = einops.repeat(gmm.get_centers(), "N C3 -> B N C3", B=B) + shift
        rot_centers = einops.einsum(rot_mats, centers, "B C31 C32, B N C32 -> B N C31")

        sigmas = einops.rearrange(gmm.get_sigmas(), "N -> 1 1 N")
        sigmas = 2 * sigmas**2
        proj_x = einops.rearrange(self.line_coords, "D -> 1 D 1") \
                 - einops.rearrange(rot_centers[..., 0], "B N -> B 1 N")
        proj_x = torch.exp(-proj_x**2 / sigmas)
        proj_y = einops.rearrange(self.line_coords, "D -> 1 D 1") \
                 - einops.rearrange(rot_centers[..., 1], "B N -> B 1 N")
        proj_y = torch.exp(-proj_y**2 / sigmas)
        proj = einops.einsum(proj_x, gmm.get_amplitudes(), proj_y, "B D1 N, N, B D2 N -> B D1 D2")
        proj = einops.rearrange(proj, "B NX NY -> B NY NX")
        return proj

    def project_slow(self, gmm: AbsGMM, rot_mats: torch.FloatTensor, shift: torch.FloatTensor):
        # batch_size: each shift is binded with a rotation matrix
        ASSERT_SHAPE(rot_mats, (None, 3, 3))
        ASSERT_SHAPE(shift, (None, None, 3))
        assert shift.shape[0] == rot_mats.shape[0]

        return batch_e2gmm_projection(gmm.get_centers().unsqueeze(0) + shift, gmm.get_sigmas(), gmm.get_amplitudes(),
                                      rot_mats, self.plane_coords, self.plane_shape)


class AtomGMMProjector(ProjectorBase):

    def density(self, gmm, xyz, q_vec, alpha, scale_translation=1.):
        centers = gmm.get_transformed_centers(q_vec=q_vec, alpha=alpha, scale_translation=scale_translation)
        delta = xyz[None, :, None, :] - centers[:, None, :, :]  # (batch_size, N_points, N_centers, 3)
        p = gmm.get_amplitudes() * torch.exp(-(delta**2).sum(-1) /
                                             (2 * gmm.get_sigmas()**2))  # (batch_size, N_points, N_centers)
        p = p.sum(-1)  # (batch_size, N_points, )
        return p

    def project(self, gmm, rot_mats, q_vec, alpha, scale_translation=1.):
        centers = gmm.get_transformed_centers(q_vec=q_vec, alpha=alpha, scale_translation=scale_translation)
        rot_centers = einops.einsum(rot_mats, centers, "B C31 C32, B N C32 -> B N C31")

        sigmas = einops.rearrange(gmm.get_sigmas(), "N -> 1 1 N")
        sigmas = 2 * sigmas**2
        proj_x = einops.rearrange(self.line_coords, "D -> 1 D 1") \
                 - einops.rearrange(rot_centers[..., 0], "B N -> B 1 N")
        proj_x = torch.exp(-proj_x**2 / sigmas)
        proj_y = einops.rearrange(self.line_coords, "D -> 1 D 1") \
                 - einops.rearrange(rot_centers[..., 1], "B N -> B 1 N")
        proj_y = torch.exp(-proj_y**2 / sigmas)
        proj = einops.einsum(proj_x, proj_y, gmm.get_amplitudes(), "B D1 N, B D2 N, N -> B D1 D2")
        proj = einops.rearrange(proj, "B NX NY -> B NY NX")
        return proj
