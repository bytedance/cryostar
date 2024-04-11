from typing import Optional

import einops
import torch
import torch.nn.functional as F


def shift_coords(coords: torch.Tensor,
                 x_half_range: float,
                 y_half_range: float,
                 z_half_range: float,
                 Nx: int,
                 Ny: int,
                 Nz: Optional[int],
                 flip: bool = False):
    """
    Shifts the coordinates and puts the DC component at (0, 0, 0).
    
    If an image has an even size, such as 4, the `fftshift`ed freq-map looks like:
    `[-0.50, -0.25,  0.00,  0.25]`, which means the DC term locates at the bottom-right
    block. It has an offset compared with uniformly initialized coords:
    `[-0.50, -0.17,  0.17,  0.50]`. The offset is a left-shift of 0.17, i.e.,
    `half-range/(num_grids-1)`.

    This may cause errors when we apply a frequency-based operation to the image 
    in the fourier domain, such as translation. See `FourierGridTranslate` for details.

    See Also: https://pytorch.org/docs/stable/generated/torch.fft.fftshift.html

    Input:
        coords:     torch.Tensor (..., 3 or 2)
        x_range:    float, (max_x - min_x) / 2
        y_range:    float, (max_y - min_y) / 2
        z_range:    float, (max_z - min_z) / 2
        Nx:         int 
        Ny:         int
        Nz:         int
        flip:       bool

    Returns:
        coords: torch.Tensor (..., 3 or 2)
    """
    alpha = -1.
    if flip:  # "unshift" the coordinates.
        alpha = 1.

    if Nx % 2 == 0:
        x_shift = coords[..., 0] + alpha * x_half_range / (Nx - 1)
    else:
        x_shift = coords[..., 0]
    if Ny % 2 == 0:
        y_shift = coords[..., 1] + alpha * y_half_range / (Ny - 1)
    else:
        y_shift = coords[..., 1]
    if Nz is not None:
        if Nz % 2 == 0:
            z_shift = coords[..., 2] + alpha * z_half_range / (Nz - 1)
        else:
            z_shift = coords[..., 2]
        coords = torch.cat((x_shift.unsqueeze(-1), y_shift.unsqueeze(-1), z_shift.unsqueeze(-1)), dim=-1)
    else:
        coords = torch.cat((x_shift.unsqueeze(-1), y_shift.unsqueeze(-1)), dim=-1)
    return coords


class FourierDownsample(torch.nn.Module):
    def __init__(self, L, D, device=None) -> None:
        super().__init__()
        self.L = L
        self.D = D

    def transform(self, f_images: torch.Tensor):
        """

            Input:
                f_images: (B, NY, NX)

            Returns:
                f_img_down: (B, NY', NX')
        """
        B, NY, NX = f_images.shape
        assert self.D == NY == NX
        assert self.L % 2 == 0
        assert self.L <= self.D
        if NY == self.L and NX == self.L:
            return f_images

        # follow cryostar.util.fft_utils
        DC_loc = self.D // 2
        DC_left = self.L // 2
        DC_right = self.L - DC_left

        s = DC_loc - DC_left
        e = DC_loc + DC_right

        f_img_down = f_images[..., s:e, s:e]

        return f_img_down


class SpatialGridDownsample(torch.nn.Module):

    def __init__(self, L, D, device=None) -> None:
        super().__init__()
        self.L = L
        self.D = D
        # yapf: disable
        coords = torch.stack(torch.meshgrid([
            torch.linspace(-1.0, 1.0, self.L, device=device),
            torch.linspace(-1.0, 1.0, self.L, device=device)],
        indexing="xy"), dim=-1) # (D, D, 2)
        # yapf: enable
        self.register_buffer("coords", coords)

    def transform(self, images: torch.Tensor):
        """

            Input:
                images: (B, NY, NX)

            Returns:
                images: (B, NY', NX')
        """
        B, NY, NX = images.shape
        assert self.D == NY == NX
        if NY == self.L and NX == self.L:
            return images
        # yapf: disable
        img_down = F.grid_sample(
            einops.repeat(images, "B NY NX -> B 1 NY NX"),
            einops.repeat(self.coords, "NY NX C2 -> B NY NX C2", B=B),
        align_corners=True)
        # yapf: enable
        img_down = einops.rearrange(img_down, "B 1 NY NX -> B NY NX")
        return img_down


class FourierGridTranslate(torch.nn.Module):
    """
        DFT's translation is:
            `f(x - x0, y - y0) <=> F(u, v) exp(-2 j \pi (x0 u + y0 v) / N )`
        where `x, y, u, v` all have a range of `N`, so `(x0 u + y0 v) / N \in (0, N)`

        Here we initialize the `u, v` coordinates between `(-0.5, 0.5)` so that the 
        range is 1, where the `1/N` term can be ignored.

        See also: https://dsp.stackexchange.com/questions/40228/translation-property-of-2-d-discrete-fourier-transform

        Important notes:
            If `N=4`, the coordinates u will be `[-0.5, -0.17, 0.17, 0.5]`, but the 
            `fft`ed image's frequency is `[-0.50, -0.25, 0.00, 0.25]`, so we have to 
            add some corrections:
                - right-shift `u` to be `[-0.50, -0.25, 0.00, 0.25]`
                - perform multiplication

    """

    def __init__(self, D, device=None) -> None:
        super().__init__()
        self.D = D
        # yapf: disable
        coords = torch.stack(torch.meshgrid([
            torch.linspace(-1.0, 1.0, self.D, device=device),
            torch.linspace(-1.0, 1.0, self.D, device=device)],
        indexing="ij"), dim=-1).reshape(-1, 2) / 2
        # yapf: enable
        coords = shift_coords(coords, 0.5, 0.5, None, self.D, self.D, None, False)
        self.register_buffer("coords", coords)

    def transform(self, images: torch.Tensor, trans: torch.Tensor):
        """
            The `images` are stored in `YX` mode, so the `trans` is also `YX`!

            Input:
                images: (B, NY, NX)
                trans:  (B, T,  2)

            Returns:
                images: (B, T,  NY, NX)
        """
        B, NY, NX = images.shape
        assert self.D == NY == NX
        assert images.shape[0] == trans.shape[0]
        images = einops.rearrange(images, "B NY NX -> B 1 (NY NX)")
        delta = trans @ self.coords.t() * -2j * torch.pi
        images_trans = torch.exp(delta) * images
        images_trans = einops.rearrange(images_trans, "B T (NY NX) -> B T NY NX", NY=self.D, NX=self.D)
        return images_trans


class SpatialGridTranslate(torch.nn.Module):

    def __init__(self, D, device=None) -> None:
        super().__init__()
        self.D = D
        # yapf: disable
        coords = torch.stack(torch.meshgrid([
            torch.linspace(-1.0, 1.0, self.D, device=device),
            torch.linspace(-1.0, 1.0, self.D, device=device)],
        indexing="ij"), dim=-1).reshape(-1, 2)
        # yapf: enable
        self.register_buffer("coords", coords)

    def transform(self, images: torch.Tensor, trans: torch.Tensor):
        """
            The `images` are stored in `YX` mode, so the `trans` is also `YX`!

            Supposing that D is 96, a point is at 0.0:
                - adding 48 should move it to the right corner which is 1.0
                    1.0 = 0.0 + 48 / (96 / 2)
                - adding 96(>48) should leave it at 0.0
                    0.0 = 0.0 + 96 / (96 / 2) - 2.0
                - adding -96(<48) should leave it at 0.0
                    0.0 = 0.0 - 96 / (96 / 2) + 2.0

            Input:
                images: (B, NY, NX)
                trans:  (B, T,  2)

            Returns:
                images: (B, T,  NY, NX)
        """
        B, NY, NX = images.shape
        assert self.D == NY == NX
        assert images.shape[0] == trans.shape[0]

        grid = einops.rearrange(self.coords, "N C2 -> 1 1 N C2") - \
            einops.rearrange(trans, "B T C2 -> B T 1 C2") * 2 / self.D
        grid = grid.flip(-1)  # convert the first axis from slow-axis to fast-axis
        grid[grid >= 1] -= 2
        grid[grid <= -1] += 2
        grid.clamp_(-1.0, 1.0)

        sampled = F.grid_sample(einops.rearrange(images, "B NY NX -> B 1 NY NX"), grid, align_corners=True)

        sampled = einops.rearrange(sampled, "B 1 T (NY NX) -> B T NY NX", NX=NX, NY=NY)
        return sampled


class GridRotate(torch.nn.Module):

    def __init__(self, D, device=None) -> None:
        super().__init__()
        self.D = D
        # yapf: disable
        coords = torch.stack(torch.meshgrid([
            torch.linspace(-1.0, 1.0, self.D, device=device),
            torch.linspace(-1.0, 1.0, self.D, device=device)],
        indexing="xy"), dim=-1) # (D, D, 2)
        # yapf: enable
        self.register_buffer("coords", coords)

    def transform(self, images: torch.Tensor, angles: torch.Tensor):
        """
            Rotate an image by angles anti-clockwisely.
            
            FIXME: maybe bugs when rotate an even-sized Fourier image.

            Input:
                images: (B, NY, NX)
                angles: (B, Q)

            Returns:
                images: (B, Q,  NY, NX)
        """
        B, NY, NX = images.shape
        B, Q = angles.shape
        assert self.D == NY == NX
        assert images.shape[0] == angles.shape[0]

        img_expanded = einops.rearrange(images, "B NY NX -> B 1 NY NX")

        cos = torch.cos(angles)
        sin = torch.sin(angles)
        rot = einops.rearrange([cos, -sin, sin, cos], "(C2_1 C2_2) B Q-> B Q C2_1 C2_2", C2_1=2, C2_2=2)

        grid = einops.einsum(self.coords, rot, "NY NX C2_1, B Q C2_1 C2_2 -> B Q NY NX C2_2")
        grid = einops.rearrange(grid, "B Q NY NX C2 -> B (Q NY) NX C2")

        img_rotated = F.grid_sample(img_expanded, grid, align_corners=True)
        img_rotated = einops.rearrange(img_rotated, "B 1 (Q NY) NX -> B Q NY NX", B=B, Q=Q)
        return img_rotated


class FourierGridProjector(torch.nn.Module):
    """
        For downsampling, we only need to crop around the DC term in the fourier space, 
        equivilant to grid sampling in the spatial space. It is easy to crop 
        `[-2, -1, 0, 1, 2]` to `[-1, 0, 1]`, or crop `[-3, -1, 1, 3]` to `[-1, 1]`, 
        so make sure that `D` and `L` are both even or odd to avoid unnecessary 
        interpolation, which may cause numercial issues.

        Important notes:
            We perform `shift_coords` for two times to bridge the `inconsistency` between
            `grid_sample` and `fftshift` when the side shape is an even number. 
            If D is 4, the 1D coordinates we generate for `grid_sample` is 
            `[-1.0, -0.33, 0.33, 1.0]`. However, the `fft`ed image's DC term will locate 
            at the bottom-right block, i.e., the coordinate is `(0.33, 0.33)`.
                - First `shift_coords`: When rotating the coordinates, we do not want the DC 
                    term to be rotated. so we have to shift the grid to ensure that the DC 
                    term's coordinate is `(0, 0)`.
                - Second `shift_coords`: After rotations, the DC term's coordinate is `(0, 0)`. 
                    When we do `grid_sample`, we want to ensure that the DC term is still
                    `(0.33, 0.33)` so that it will not be interpolated.
            Maybe another solution is directly substracting the DC term's 
            cooridnate by `grid=grid-grid[2][2]`.

    """

    def __init__(self, L, D, device=None) -> None:
        super().__init__()
        self.L = L
        self.D = D
        assert self.L % 2 == self.D % 2, \
            "The original size and the projection size should be both odd or even"
        linspace = torch.linspace(-L / D, L / D, L, device=device)
        xx, yy = torch.meshgrid([linspace, linspace], indexing="ij")
        coords = torch.stack([xx, yy, torch.zeros_like(xx)], dim=-1)  # (L, L, 3)
        coords = coords.unsqueeze(2)  # (L, L, 1, 3)
        coords = shift_coords(coords, L / D, L / D, 0.0, self.L, self.L, 1, flip=False)
        self.register_buffer("coords", coords)

    def transform(self, density: torch.Tensor, rotation: torch.Tensor):
        """
            Input:
                density: (B, C=2, NZ, NY, NX)
                rotation: (B, Q,  3,  3)

            Returns:
                images: (B, Q, C=2, NY, NX)
        """
        B, C, NZ, NY, NX = density.shape
        B, Q, _, _ = rotation.shape
        assert self.D == NZ == NY == NX
        assert density.shape[0] == rotation.shape[0]

        # expanded_density = einops.repeat(density, "B C NZ NY NX -> B Q C NZ NY NX", Q=Q)
        # expanded_density = einops.rearrange(expanded_density, "B Q C NZ NY NX -> (B Q) C NZ NY NX")
        # expanded_rotation = einops.rearrange(rotation, "B Q C31 C32 -> (B Q) C31 C32")
        # rot_coords = torch.einsum("ijkn, bnm -> bijkm", self.coords, expanded_rotation)
        # rot_coords = shift_coords(rot_coords, 1.0, 1.0, 0.0, self.D, self.D, self.D, flip=True)
        # fproj = F.grid_sample(expanded_density, rot_coords, align_corners=True)
        # fproj = einops.rearrange(fproj, "(B Q) C X Y 1 -> B Q C Y X", B=B, Q=Q).contiguous()

        coords = einops.repeat(self.coords, "NX NY 1 C3 -> B Q NX NY 1 C3", B=B, Q=Q)
        coords = einops.rearrange(coords, "B Q NX NY 1 C3 -> B NX NY Q C3")
        coords = einops.einsum(
            coords,
            rotation,
            "B NX NY Q C3_1, B Q C3_1 C3_2 -> B NX NY Q C3_2",
        )
        coords = shift_coords(coords, self.L / self.D, self.L / self.D, 1.0, self.L, self.L, self.L, flip=True)
        fproj = F.grid_sample(density, coords, align_corners=True)
        fproj = einops.rearrange(fproj, "B C NX NY Q-> B Q C NY NX", B=B, Q=Q).contiguous()

        return fproj


class SpatialGridProjector(torch.nn.Module):

    def __init__(self, L, D, device=None) -> None:
        super().__init__()
        self.L = L
        self.D = D
        linspace = torch.linspace(-1, 1, L, device=device)
        xx, yy, zz = torch.meshgrid([linspace, linspace, linspace], indexing="ij")
        coords = torch.stack([xx, yy, zz], dim=-1)  # (L, L, L, 3)
        self.register_buffer("coords", coords)

    def transform(self, density: torch.Tensor, rotation: torch.Tensor):
        """
            Input:
                density: (B, NZ, NY, NX)
                rotation: (B, Q,  3,  3)

            Returns:
                images: (B, Q, NY, NX)
        """
        B, NZ, NY, NX = density.shape
        B, Q, _, _ = rotation.shape
        assert self.D == NZ == NY == NX
        assert density.shape[0] == rotation.shape[0]

        expanded_density = einops.repeat(density, "B NZ NY NX -> B Q C NZ NY NX", Q=Q, C=1)
        expanded_density = einops.rearrange(expanded_density, "B Q 1 NZ NY NX -> (B Q) 1 NZ NY NX")

        expanded_rotation = einops.rearrange(rotation, "B Q C31 C32 -> (B Q) C31 C32")
        rot_coords = torch.einsum("ijkn, bnm -> bijkm", self.coords, expanded_rotation)
        sampled = F.grid_sample(expanded_density, rot_coords, align_corners=True)
        sampled = einops.rearrange(sampled, "(B Q) 1 X Y Z -> B Q Y X Z", B=B, Q=Q)
        proj = einops.reduce(sampled, "B Q Y X Z -> B Q Y X", reduction="sum")
        return proj


class Deformer(torch.nn.Module):

    def __init__(self, D, device=None) -> None:
        super().__init__()
        self.D = D
        linspace = torch.linspace(-1, 1, D, device=device)
        xx, yy, zz = torch.meshgrid([linspace, linspace, linspace], indexing="ij")
        coords = torch.stack([xx, yy, zz], dim=-1)  # (D, D, D, 3)
        self.resizer = SpatialVolumeResizer(D)
        self.register_buffer("coords", coords)

    def transform(self, density: torch.Tensor, deformation: torch.Tensor, is_relative: bool = True):
        """
            Input:
                density: (NZ, NY, NX)

                deformation: (B, NZ, NY, NX, 3)

                    It is saved in the D-H-W shape (NZ, NY, NX, ...) and x-y-z order (..., 3)

                is_relative: bool

                    Is the deformation a relative coordinate, i.e. an offset to the grid, 
                    or an absolute coordinate, i.e., an offset to the origin (0, 0, 0)?

            Returns:
                images: (B, NZ, NY, NX)
            
            Notes:

                To be consistent with `SpatialGridProjector`, the `coords` (LxLxLx3) is saved in `D-H-W` 
                shape (the first 3 dimensions) but in `z-y-x` order (the last dimension). 
                The `grid_sample` function requires the grid to be in `x-y-z` order. I am
                not sure if the statement is rigorous, but you can easily check it by setting
                `deformation` to `zero` and finding that:
                ```
                    F.grid_sample(density, self.coords) == density.permute()
                    F.grid_sample(density, self.coords.flip(-1)) == density
                ```
        """
        NZ, NY, NX = density.shape
        B, NZ2, NY2, NX2, _ = deformation.shape
        assert self.D == NZ == NY == NX
        assert NZ2 == NY2 == NX2

        if NZ2 != self.D:
            deformation = einops.rearrange(deformation, "B NZ NY NX C3 -> B C3 NZ NY NX")
            deformation = self.resizer.transform(deformation)
            deformation = einops.rearrange(deformation, "B C3 NZ NY NX -> B NZ NY NX C3")

        if is_relative:
            deformed_coords = self.coords[None, ...].flip(-1) + deformation
        else:
            deformed_coords = deformation

        deformed_coords = einops.rearrange(deformed_coords, "B NZ NY NX C3 -> 1 (B NZ) NY NX C3")
        deformed_coords = deformed_coords

        density = einops.rearrange(density, "NZ NY NX -> 1 1 NZ NY NX")

        sampled = F.grid_sample(density, deformed_coords, align_corners=True)
        sampled = einops.rearrange(sampled, "1 1 (B NZ) NY NX -> B NZ NY NX", B=B, NZ=NZ)
        return sampled


class SpatialVolumeResizer(torch.nn.Module):

    def __init__(self, D, device=None) -> None:
        super().__init__()
        self.D = D
        linspace = torch.linspace(-1, 1, D, device=device)
        xx, yy, zz = torch.meshgrid([linspace, linspace, linspace], indexing="ij")
        coords = torch.stack([xx, yy, zz], dim=-1)  # (D, D, D, 3)
        self.register_buffer("coords", coords)

    def transform(self, density: torch.Tensor):
        """
            Input:
                density: (B, C, NZ, NY, NX)

            Returns:
                images: (B, C, NZ=D, NY=D, NX=D)

        """
        B, C, NZ, NY, NX = density.shape
        assert NZ == NY == NX
        if NZ == self.D:
            return density

        sampled = F.grid_sample(
            density,
            einops.repeat(self.coords.flip(-1), "NZ NY NX C3 -> B NZ NY NX C3", B=len(density)), align_corners=True
        )
        return sampled
