import numpy as np

import torch
from torch import nn


# 2D FFT matrix is conjugate-symmetric.
def e2_ring_masks(shape):
    """Create EMAN2 style Fourier ring mask.

    Args:
        shape: image (N x N) side shape N

    Returns:
        rings: (N, N // 2 + 1, N // 2), DC term is at upper-left
    """
    idx = np.indices((shape, shape)) - shape // 2
    idx = np.fft.ifftshift(idx)
    idx = idx[:, :, :shape // 2 + 1]

    rr = np.round(np.sqrt(np.sum(idx**2, axis=0))).astype(int)
    rings = np.zeros((shape, shape // 2 + 1, shape // 2), dtype=np.float32)
    for i in range(shape // 2):
        rings[:, :, i] = (rr == i)
    return rings


def cpx_norm(c):
    return torch.pow(c.real, 2) + torch.pow(c.imag, 2)


class FRCLoss(nn.Module):
    r"""Fourier Ring Correlation loss.
    """

    def __init__(self, shape, min_freq_id=1, max_freq_id=-1):
        super().__init__()
        self.min_freq_id = min_freq_id
        self.max_freq_id = max_freq_id

        self.register_buffer("ring_masks", e2_ring_masks(shape))

    def forward(self, a_images, b_images):
        f_a_images = torch.fft.rfft2(a_images)
        f_b_images = torch.fft.rfft2(b_images)

        a_norm = torch.tensordot(cpx_norm(f_a_images), self.ring_masks, [[-2, -1], [0, 1]])  # b, num_freq
        b_norm = torch.tensordot(cpx_norm(f_b_images), self.ring_masks, [[-2, -1], [0, 1]])

        # denominator
        den = torch.sqrt(a_norm) * torch.sqrt(b_norm)
        den = torch.maximum(den, den.new_tensor(1e-4))

        # numerator
        num = f_a_images.real * f_b_images.real + f_a_images.imag * f_b_images.imag
        frc = torch.tensordot(num, self.ring_masks, [[-2, -1], [0, 1]]) / den

        return torch.mean(frc[:, self.min_freq_id:self.max_freq_id])
