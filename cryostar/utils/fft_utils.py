import math

import torch

from cryostar.utils.misc import create_sphere_mask, create_circular_mask
"""
    Hartley Transform is defined in https://en.wikipedia.org/wiki/Hartley_transform
    where:
        H{w} = ( F{w} + F{-w} ) / 2 - i ( F{w} - F{-w} ) / 2 
        H{-w} = ( F{-w} + F{w} ) / 2 - i ( F{-w} - F{w} ) / 2 
    
    The conversion between Fourier and Hartley is:
    - Odd case:

        - Fourier -> Hartley

            Since:
                    0  1  2
                0 [        ]
            F = 1 [   DC   ]
                2 [        ]
            H[00] = (F[00] + F[22]) / 2 - i (F[00] - F[22]) / 2
                  = (2 * F[00].real) / 2 - i (i 2 * F[00].imag) / 2
                  = F[00].real + F[00].imag
            
            We have:
                H = F.real + F.imag
            
        - Hartley -> Fourier

            Since:
                H       = F.real + F.imag
                flip(H) = flip(F.real) + flip(F.imag) 
                        = F.real - F.imag

            We have: 
                F.real = (H + flip(H)) / 2
                F.imag = (H - flip(H)) / 2
        
    - Even case:

        - Fourier -> Hartley

            Since:
                    0  1  2  3
                0 [           ]
            F = 1 [           ]
                2 [       DC  ]
                3 [           ]
            We can augment F to be odd-shaped, where the last row/col is the first row/col.
            (The frequency is [-0.5, -0.25, 0.0, 0.25], where the frequency -0.5
            can also be considered as 0.5)
                     0  1  2  3  4
                 0 [ a  b  c  d  e]
                 1 [ f           f]
            F' = 2 [ g     DC    g]
                 3 [ h           h]
                 4 [ a  b  c  d  e]
            (Some properties of F:
                - F[20]=F[24] (g=g) and F[20]=F[24]* (g=g*), g is real.
                - F[00]=F[04] (a=e) and F[00]=F[44]* (a=e*), a=e is real.)

            According to the odd case,
                H' = F'.real + F'.imag
            where:    
                H'[last row] = F'[last row].real + F'[last row].imag
                             = F'[first row].real + F'[first row].imag
                             = H'[first row]

            Takeaway:
                - Convert F to F' by duplicating the first row/col.
                - Convert F' to H'
                - Drop the last row/col
        

        Hartley -> Fourier
            Take away:
                - Convert H to H' by duplicating the first row/col.
                - Convert H' to F'
                - Drop the last row/col         
"""


def hartley_to_fourier_2d(H):
    """
        Example:
            >>> f = hartley_to_fourier_2d(torch.rand(4, 4))
            >>> torch.fft.ifft2(torch.fft.ifftshift(f)).imag  # the imag part is 0
    """
    assert len(H.shape) == 2 and H.shape[0] == H.shape[1]
    D = H.shape[0]
    dims = (0, 1)

    if D % 2 == 1:
        H_flip = torch.flip(H, dims=dims)
        F = (H + H_flip) / 2 + (H - H_flip) / 2 * 1j
        return F
    else:
        H_aug = torch.zeros((D + 1, D + 1), dtype=H.dtype, device=H.device)
        H_aug[:D, :D] = H
        H_aug[D, :D] = H[0, :D]
        H_aug[:D, D] = H[:D, 0]
        H_aug[D, D] = H[0, 0]
        H_aug_flip = torch.flip(H_aug, dims=dims)
        F_aug = (H_aug + H_aug_flip) / 2 + (H_aug - H_aug_flip) / 2 * 1j
        F = F_aug[:D, :D]
        return F
    

def batch_hartley_to_fourier_2d(Hs):
    """
        A batch version of `hartley_to_fourier_2d`.
        Example:
            >>> x = torch.rand(3, 4, 4)
            >>> y1 = torch.stack(list(hartley_to_fourier_2d(x[i]) for i in range(3))) 
            >>> y2 = batch_hartley_to_fourier_2d(x)
            >>> (y1 - y2).abs().sum()
    """
    assert len(Hs.shape) == 3 and Hs.shape[1] == Hs.shape[2]
    bsz, D, _ = Hs.shape
    dims = (1, 2)

    if D % 2 == 1:
        Hs_flip = torch.flip(Hs, dims=dims)
        F = (Hs + Hs_flip) / 2 + (Hs - Hs_flip) / 2 * 1j
        return F
    else:
        Hs_aug = torch.zeros((bsz, D + 1, D + 1), dtype=Hs.dtype, device=Hs.device)
        Hs_aug[:, :D, :D] = Hs
        Hs_aug[:, D, :D] = Hs[:, 0, :D]
        Hs_aug[:, :D, D] = Hs[:, :D, 0]
        Hs_aug[:, D, D] = Hs[:, 0, 0]
        Hs_aug_flip = torch.flip(Hs_aug, dims=dims)
        Fs_aug = (Hs_aug + Hs_aug_flip) / 2 + (Hs_aug - Hs_aug_flip) / 2 * 1j
        Fs = Fs_aug[:, :D, :D]
        return Fs


def hartley_to_fourier_3d(H):
    """
        Example:
            >>> f = hartley_to_fourier_3d(torch.rand(4, 4, 4))
            >>> torch.fft.ifftn(torch.fft.ifftshift(f), dim=(-2, -1, 0)).imag  # the imag part is 0
    """
    assert len(H.shape) == 3 and H.shape[0] == H.shape[1] == H.shape[2]
    D = H.shape[0]
    dims = (0, 1, 2)

    if D % 2 == 1:
        H_flip = torch.flip(H, dims=dims)
        F = (H + H_flip) / 2 + (H - H_flip) / 2 * 1j
        return F
    else:
        H_aug = torch.zeros((D + 1, D + 1, D + 1), dtype=H.dtype, device=H.device)
        H_aug[:D, :D, :D] = H
        # set three faces
        H_aug[D, :D, :D] = H[0, :, :]
        H_aug[:D, D, :D] = H[:, 0, :]
        H_aug[:D, :D, D] = H[:, :, 0]
        # set three sides
        H_aug[:D, D, D] = H[:, 0, 0]
        H_aug[D, :D, D] = H[0, :, 0]
        H_aug[D, D, :D] = H[0, 0, :]
        # set vertex
        H_aug[D, D, D] = H[0, 0, 0]
        H_aug_flip = torch.flip(H_aug, dims=dims)
        F_aug = (H_aug + H_aug_flip) / 2 + (H_aug - H_aug_flip) / 2 * 1j
        F = F_aug[:D, :D, :D]
        return F


def _ht_3d(x):
    x = torch.fft.fftshift(x, dim=(-3, -2, -1))
    y = torch.fft.fftn(x, dim=(-3, -2, -1))
    y = torch.fft.fftshift(y, dim=(-3, -2, -1))
    return y


def real_to_ht_3d(r):
    h = _ht_3d(r)
    h = h / math.sqrt(r.shape[-1] * r.shape[-2] * r.shape[-3])
    return h.real - h.imag


def ht_to_real_3d(h):
    r = _ht_3d(h)
    r = r / math.sqrt(r.shape[-1] * r.shape[-2] * r.shape[-3])
    return r.real - r.imag


@torch.autocast("cuda")
def primal_to_fourier_2d(r: torch.Tensor) -> torch.Tensor:
    with torch.autocast("cuda", enabled=False):
        r = torch.fft.ifftshift(r.float(), dim=(-2, -1))
        f = torch.fft.fftshift(torch.fft.fftn(r, s=(r.shape[-2], r.shape[-1]), dim=(-2, -1)), dim=(-2, -1))
    return f


@torch.autocast("cuda")
def primal_to_fourier_3d(r: torch.Tensor) -> torch.Tensor:
    with torch.autocast("cuda", enabled=False):
        r = torch.fft.ifftshift(r.float(), dim=(-3, -2, -1))
        f = torch.fft.fftshift(torch.fft.fftn(r, s=(r.shape[-3], r.shape[-2], r.shape[-1]), dim=(-3, -2, -1)),
                               dim=(-3, -2, -1))
    return f


def fourier_to_primal_2d(f: torch.Tensor) -> torch.Tensor:
    f = torch.fft.ifftshift(f, dim=(-2, -1))
    return torch.fft.fftshift(torch.fft.ifftn(f, s=(f.shape[-2], f.shape[-1]), dim=(-2, -1)), dim=(-2, -1))


def fourier_to_primal_3d(r: torch.Tensor) -> torch.Tensor:
    r = torch.fft.ifftshift(r, dim=(-3, -2, -1))
    return torch.fft.fftshift(torch.fft.ifftn(r, s=(r.shape[-3], r.shape[-2], r.shape[-1]), dim=(-3, -2, -1)),
                              dim=(-3, -2, -1))


def downsample_3d(r: torch.Tensor, down_side: int) -> torch.Tensor:
    f_density = primal_to_fourier_3d(r)
    D = f_density.shape[0]

    # even: D = 8, DC_loc = 4
    # - - - - 0 + + +
    # odd: D = 7, DC_loc = 3
    # - - - 0 + + +
    DC_loc = D // 2

    # even: L = 8, [DC-4, DC+4)
    # - - - - 0 + + +
    # odd:  L = 7, [DC-3, DC+4)
    # - - - 0 + + +
    DC_left = down_side // 2
    DC_right = down_side - DC_left

    s = DC_loc - DC_left
    e = DC_loc + DC_right
    f_density_down = f_density[s:e, s:e, s:e]
    mask = torch.tensor(create_sphere_mask(down_side, down_side, down_side, radius=down_side // 2))
    f_density_down[~mask] = 0
    density_down = fourier_to_primal_3d(f_density_down).real
    return density_down


def downsample_2d(r: torch.Tensor, down_side: int) -> torch.Tensor:
    f_image = primal_to_fourier_2d(r)
    D = f_image.shape[0]
    DC_loc = D // 2
    DC_left = down_side // 2
    DC_right = down_side - DC_left

    s = DC_loc - DC_left
    e = DC_loc + DC_right
    f_image_down = f_image[s:e, s:e]
    mask = torch.tensor(create_circular_mask(down_side, down_side, radius=down_side // 2))
    f_image_down[~mask] = 0
    image_down = fourier_to_primal_2d(f_image_down).real
    return image_down
