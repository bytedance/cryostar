import math
from abc import ABCMeta, abstractmethod

import numpy as np
import scipy.constants
import torch
import torch.nn.functional as F
from torch import nn
from cryostar.utils.fft_utils import primal_to_fourier_2d, fourier_to_primal_2d
from cryostar.utils.ctf import compute_ctf


class CTFBase(nn.Module, metaclass=ABCMeta):
    """
    CTF abstract class
    """

    def __init__(self, resolution, num_particles, requires_grad=False):
        super(CTFBase, self).__init__()
        self.resolution = resolution
        self.requires_grad = requires_grad

    @abstractmethod
    def forward(self, x_fourier, idcs=0, ctf_params={}, mode='gt', frequency_marcher=None):
        ...


class CTFIdentity(CTFBase):

    def __init__(self, resolution=.8, num_particles=500):
        """
        Initialization of a CTF that does nothing.

        Parameters
        ----------
        resolution: float
        num_particles: int
        """
        super().__init__(resolution, num_particles, False)

    def forward(self, x_fourier, idcs=0, ctf_params=None, mode='none', frequency_marcher=None):
        if x_fourier.dim() == 3:
            x_fourier = x_fourier[None, ...]  # adds a batch dimension so as to be compatible with our other CTFs
        if ctf_params is None:
            ctf_params = {}
        return x_fourier


class CTFRelion(CTFBase):

    def __init__(self,
                 size=257,
                 resolution=0.8,
                 kV=300.0,
                 valueNyquist=1.,
                 defocusU=1.,
                 defocusV=1.,
                 angleAstigmatism=0.,
                 cs=2.7,
                 phasePlate=0.,
                 amplitudeContrast=.1,
                 bFactor=0.,
                 num_particles=500,
                 requires_grad=False,
                 precompute=False,
                 flip_images=False):
        super(CTFRelion, self).__init__(resolution, num_particles, requires_grad)
        self.requires_grad = requires_grad
        self.flip_images = flip_images

        self.size = size  # in pixel
        self.resolution = resolution  # in angstrom
        self.kV = kV  # in kilovolt

        self.valueNyquist = valueNyquist
        self.phasePlate = phasePlate / 180. * np.pi  # in radians (converted from degrees)
        self.amplitudeContrast = amplitudeContrast
        self.bFactor = bFactor

        self.frequency = 1. / self.resolution

        self.wavelength = self._get_ewavelength(self.kV * 1e3)  # input in V (so we convert kv*1e3)

        angleAstigmatism = angleAstigmatism / 180. * np.pi  # input in degree converted in radian
        cs = cs * 1e7  # input in mm converted in angstrom
        # the angleAstigmatism, defocusU, defocusV and cs are nn.Parameter of size (N, 1, 1)
        self.angleAstigmatism = nn.Parameter(angleAstigmatism * torch.ones((num_particles, 1, 1), dtype=torch.float32),
                                             requires_grad=requires_grad)
        self.cs = nn.Parameter(cs * torch.ones((num_particles, 1, 1), dtype=torch.float32), requires_grad=requires_grad)
        self.defocusU = nn.Parameter(defocusU * torch.ones((num_particles, 1, 1), dtype=torch.float32),
                                     requires_grad=requires_grad)
        self.defocusV = nn.Parameter(defocusV * torch.ones((num_particles, 1, 1), dtype=torch.float32),
                                     requires_grad=requires_grad)

        self.precomputed_filters = precompute

        ax = torch.linspace(-1. / (2. * resolution), 1 / (2. * resolution), self.size)
        mx, my = torch.meshgrid(ax, ax, indexing="ij")
        self.register_buffer("r2", mx**2 + my**2)
        self.register_buffer("r", torch.sqrt(self.r2))
        self.register_buffer("angleFrequency", torch.atan2(my, mx))

        if not self.requires_grad and self.precomputed_filters:
            print("Precomputing hFourier in CTF")
            self.register_buffer('hFourier', self.get_ctf(torch.arange(num_particles), num_particles))

    def _get_ewavelength(self, U):
        # assumes V as input, returns wavelength in angstrom
        h = scipy.constants.h
        e = scipy.constants.e
        c = scipy.constants.c
        m0 = scipy.constants.m_e

        return h / math.sqrt(2. * m0 * e * U) / math.sqrt(1 + e * U / (2 * m0 * c**2)) * 1e10

    def get_ctf(self, idcs, B, cpu_params={}, frequency_marcher=None):
        defocusU = self.defocusU[idcs, :, :]
        defocusV = self.defocusV[idcs, :, :]
        angleAstigmatism = self.angleAstigmatism[idcs, :, :]
        cs = self.cs[idcs, :, :]

        ac = self.amplitudeContrast
        pc = math.sqrt(1. - ac**2)
        K1 = np.pi / 2. * cs * self.wavelength**3
        K2 = np.pi * self.wavelength

        # Cut-off from frequency marcher
        if frequency_marcher is not None:
            self.size_after_fm = 2 * frequency_marcher.f + 1
            if self.size_after_fm > self.size:
                self.size_after_fm = self.size
            angleFrequency = frequency_marcher.cut_coords_plane(self.angleFrequency.reshape(
                self.size, self.size, 1)).reshape(self.size_after_fm, self.size_after_fm)
            r2 = frequency_marcher.cut_coords_plane(self.r2.reshape(self.size, self.size,
                                                                    1)).reshape(self.size_after_fm, self.size_after_fm)
        else:
            self.size_after_fm = self.size
            angleFrequency = self.angleFrequency
            r2 = self.r2

        angle = angleFrequency - angleAstigmatism
        local_defocus = 1e4 * (defocusU + defocusV) / 2. + angleAstigmatism * torch.cos(2. * angle)

        gamma = K1 * r2**2 - K2 * r2 * local_defocus - self.phasePlate
        hFourier = -pc * torch.sin(gamma) + ac * torch.cos(gamma)

        if self.valueNyquist != 1:
            decay = np.sqrt(-np.log(self.valueNyquist)) * 2. * self.resolution
            envelope = torch.exp(-self.frequency * decay**2 * r2)
            hFourier *= envelope

        return hFourier

    def oversample_multiply_crop(self, x_fourier, hFourier):
        # we assume that the shape of the CTF is always going to be bigger
        # than the size of the input image
        input_sz = x_fourier.shape[-1]
        if input_sz != self.size_after_fm:
            x_primal = fourier_to_primal_2d(x_fourier)

            pad_len = (self.size_after_fm - x_fourier.shape[-1]) // 2  # here we assume even lengths
            p2d = (pad_len, pad_len, pad_len, pad_len)
            x_primal_padded = F.pad(x_primal, p2d, 'constant', 0)

            x_fourier_padded = primal_to_fourier_2d(x_primal_padded)

            x_fourier_padded_filtered = x_fourier_padded * hFourier[:, None, :, :]
            return x_fourier_padded_filtered[..., pad_len:-pad_len, pad_len:-pad_len]
        else:
            return x_fourier * hFourier[:, None, :, :]

    def get_cpu_params(self, idcs, ctf_params, flip=False):
        batch_size = idcs.shape[0]
        self.defocusU[idcs, :, :] = ctf_params['defocusU'][:batch_size] if not flip else\
            ctf_params['defocusU'][batch_size:]
        self.defocusV[idcs, :, :] = ctf_params['defocusV'][:batch_size] if not flip else\
            ctf_params['defocusV'][batch_size:]
        self.angleAstigmatism[idcs, :, :] = ctf_params['angleAstigmatism'][:batch_size] if not flip else\
            ctf_params['angleAstigmatism'][batch_size:]
        cpu_params = {}
        return cpu_params

    def forward(self, x_fourier, idcs=0, ctf_params={}, mode='gt', frequency_marcher=None):
        # This is when we want to prescribe parameters for the CTF
        if x_fourier.dim() == 3:
            x_fourier = x_fourier[None, ...]
        # x_fourier: B, 1, S, S
        batch_size = len(idcs)
        cpu_params = {}
        if ctf_params:
            cpu_params = self.get_cpu_params(idcs, ctf_params, flip=False)

        # if new params for the CTF have been prescribed or we are optimizing it
        # then request the evaluation of the CTF
        if not ctf_params and self.precomputed_filters and not self.requires_grad:
            hFourier = self.hFourier[idcs, :, :]
        else:
            hFourier = self.get_ctf(idcs, batch_size, cpu_params=cpu_params, frequency_marcher=frequency_marcher)

        if self.flip_images:
            flipped_hFourier = torch.flip(hFourier, [1, 2])

            hFourier = torch.cat([hFourier, flipped_hFourier], dim=0)

        return self.oversample_multiply_crop(x_fourier, hFourier)


class CTFCryoDRGN(CTFBase):

    def __init__(self,
                 size,
                 resolution,
                 num_particles=None,
                 kV=300,
                 cs=2.0,
                 amplitudeContrast=0.1,
                 requires_grad=False):
        super(CTFBase, self).__init__()
        self.size = size
        self.resolution = resolution
        self.requires_grad = requires_grad
        self.kV = kV
        self.cs = cs
        self.ac = amplitudeContrast
        ax = torch.linspace(-1. / (2. * resolution), 1 / (2. * resolution), self.size)
        mx, my = torch.meshgrid(ax, ax, indexing="ij")
        freqs = torch.stack([mx.flatten(), my.flatten()], 1)
        self.register_buffer("freqs", freqs)

    def get_ctf(self, ctf_params={}):
        bsz = len(ctf_params["defocusU"])
        device = self.freqs.device
        hFourier = compute_ctf(freqs=self.freqs.repeat(bsz, 1, 1),
                               dfu=(ctf_params["defocusU"] * 1e4).squeeze(1),
                               dfv=(ctf_params["defocusV"] * 1e4).squeeze(1),
                               dfang=torch.rad2deg(ctf_params["angleAstigmatism"]).squeeze(1),
                               volt=torch.tensor(self.kV, device=device).repeat(bsz, 1),
                               cs=torch.tensor(self.cs, device=device).repeat(bsz, 1),
                               w=torch.tensor(self.ac, device=device).repeat(bsz,
                                                                             1)).reshape(bsz, self.size, self.size)
        return hFourier

    def forward(self, x_fourier, idcs=0, ctf_params={}, mode='gt', frequency_marcher=None):
        hFourier = -self.get_ctf(ctf_params)
        return x_fourier * hFourier[:, None, :, :]
