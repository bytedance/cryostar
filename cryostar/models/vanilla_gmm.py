from typing import Union

import torch

from cryostar.utils.misc import ASSERT_SHAPE

from .gmm import AbsGMM


class VanillaGMM(AbsGMM):

    def __init__(
        self,
        *,
        centers: torch.FloatTensor,
        amplitudes: Union[None, float, torch.FloatTensor] = None,
        sigmas: Union[None, float, torch.FloatTensor] = None,
    ) -> None:
        super().__init__()
        ASSERT_SHAPE(centers, (None, 3))

        if amplitudes is None:
            amplitudes = torch.ones(size=(1, ), device=centers.device)
        elif isinstance(amplitudes, float):
            amplitudes = torch.ones(size=(1, ), device=centers.device) * amplitudes
        elif isinstance(amplitudes, torch.FloatTensor):
            ASSERT_SHAPE(amplitudes, (None, ))
            assert amplitudes.shape[0] == centers.shape[0]

        if sigmas is None:
            sigmas = torch.ones(size=(1, ), device=centers.device)
        elif isinstance(sigmas, float):
            sigmas = torch.ones(size=(1, ), device=centers.device) * sigmas
        elif isinstance(sigmas, torch.FloatTensor):
            ASSERT_SHAPE(sigmas, (None, ))
            assert sigmas.shape[0] == centers.shape[0]

        self.centers = torch.nn.Parameter(centers.clone())
        self.amplitudes = torch.nn.Parameter(amplitudes.clone())
        self.sigmas = torch.nn.Parameter(sigmas.clone())

    def set_tunable(self, *, centers=True, amplitudes=True, sigmas=True):
        self.centers.requires_grad_(centers)
        self.amplitudes.requires_grad_(amplitudes)
        self.sigmas.requires_grad_(sigmas)

    def get_centers(self):
        return self.centers

    def get_amplitudes(self):
        if self.amplitudes.shape[0] == 1:
            return self.amplitudes.repeat(len(self.centers))
        else:
            return self.amplitudes

    def get_sigmas(self):
        if self.sigmas.shape[0] == 1:
            return self.sigmas.repeat(len(self.centers))
        else:
            return self.sigmas
