import torch
from torch import nn
import einops
from cryostar.utils.transforms import shift_coords
from cryostar.nerf.ml_modules import FourierNet
from cryostar.utils.misc import create_sphere_mask, create_circular_mask
from cryostar.utils.fft_utils import batch_hartley_to_fourier_2d, hartley_to_fourier_3d, fourier_to_primal_3d


class ImplicitFourierVolume(nn.Module):

    def __init__(self, z_dim, img_sz, mask_rad, params_implicit):
        """
        Initialization of an implicit representation of the volume in Fourier space.

        Parameters
        ----------
        img_sz: int
        params_implicit: dictionary
        """
        super().__init__()
        self.img_sz = img_sz
        self.z_dim = z_dim

        lincoords = torch.linspace(-1., 1., self.img_sz)
        [X, Y] = torch.meshgrid([lincoords, lincoords], indexing="ij")
        coords = torch.stack([Y, X, torch.zeros_like(X)], dim=-1)
        coords = shift_coords(coords, 1., 1., 0, img_sz, img_sz, 1)
        self.register_buffer('plane_coords', coords.reshape(-1, 3))

        self.mask_rad = mask_rad
        if self.mask_rad != 1:
            mask = create_circular_mask(img_sz, img_sz, None, self.mask_rad / 2 * img_sz)
            plane_window_mask = torch.from_numpy(mask).reshape(-1)
            self.register_buffer('plane_window_mask', plane_window_mask)
        sphere_mask = torch.from_numpy(
            create_sphere_mask(self.img_sz, self.img_sz, self.img_sz, radius=self.mask_rad / 2 * self.img_sz)
        )
        self.register_buffer("sphere_mask", sphere_mask)

        lincoords = torch.linspace(-1., 1., self.img_sz)
        [X, Y, Z] = torch.meshgrid([lincoords, lincoords, lincoords], indexing="ij")
        coords = torch.stack([Z, Y, X], dim=-1)
        coords = shift_coords(coords, 1., 1., 1., img_sz, img_sz, img_sz)
        self.register_buffer('coords_3d', coords.reshape(-1, 3))

        self.fvol = FourierNet(net_type=params_implicit["net_type"],
                               z_dim=z_dim,
                               pe_dim=params_implicit["pe_dim"],
                               pe_type=params_implicit["pe_type"],
                               D=params_implicit["D"],
                               hidden_dim=params_implicit["hidden"],
                               force_symmetry=params_implicit['force_symmetry'])

    def forward(self, z, rotmat):
        """
        Generates a slice in Fourier space from a rotation matrix.

        Parameters
        ----------
        rotmat: torch.Tensor (B, 3, 3)

        Returns
        -------
        fplane: torch.Tensor (B, 1, img_sz, img_sz) (complex)
        """
        if self.z_dim == 0:
            assert z is None
        batch_sz = rotmat.shape[0]

        with torch.autocast("cuda", enabled=False):
            assert self.plane_coords.dtype == torch.float32
            assert rotmat.dtype == torch.float32
            rot_plane_coords = torch.bmm(self.plane_coords.repeat(batch_sz, 1, 1), rotmat)  # B, img_sz^2, 3

        if self.mask_rad != 1:
            coords_mask = einops.repeat(self.plane_window_mask, "num_coords -> bsz num_coords c3", bsz=batch_sz, c3=3)
            rot_plane_coords = rot_plane_coords[coords_mask].reshape(batch_sz, -1, 3)  # B, mask_num, 3

        fplane = self.fvol(z, rot_plane_coords)   # B, _, 1/2

        if self.mask_rad != 1:
            unmask_fplane = fplane.new_zeros(batch_sz, self.img_sz * self.img_sz, self.fvol.out_features)
            value_mask = einops.repeat(self.plane_window_mask, "num_coords -> bsz num_coords c", bsz=batch_sz, c=self.fvol.out_features)
            unmask_fplane[value_mask] = fplane.reshape(-1)
            fplane = unmask_fplane.reshape(batch_sz, self.img_sz, self.img_sz, self.fvol.out_features)
        else:
            fplane = fplane.reshape(batch_sz, self.img_sz, self.img_sz, self.fvol.out_features)

        if self.fvol.out_features == 2:
            fplane = torch.view_as_complex(fplane)   # B, img_sz, img_sz
        else:
            fplane = batch_hartley_to_fourier_2d(fplane.squeeze(-1))    # B, img_sz, img_sz

        fplane = fplane[:, None, :, :]
        return fplane

    def make_volume(self, z):
        with torch.no_grad():
            with torch.autocast("cuda", enabled=False):
                coords = self.coords_3d.unsqueeze(0)
                num_coords = coords.shape[1]
                chunk_size = 128**2 * 32
                exp_fvol = []
                for sid in range(0, num_coords, chunk_size):
                    eid = sid + chunk_size
                    exp_fvol.append(self.fvol(z, coords[:, sid:eid]))
                exp_fvol = torch.cat(exp_fvol, dim=1)
                if self.fvol.out_features == 2:
                    exp_fvol = exp_fvol.reshape(self.img_sz, self.img_sz, self.img_sz, 2)
                    exp_fvol = torch.view_as_complex(exp_fvol)
                else:
                    exp_fvol = exp_fvol.reshape(self.img_sz, self.img_sz, self.img_sz)
                    exp_fvol = hartley_to_fourier_3d(exp_fvol)

                exp_fvol[~self.sphere_mask] = 0.0
                exp_vol = fourier_to_primal_3d(exp_fvol).real
        return exp_vol
