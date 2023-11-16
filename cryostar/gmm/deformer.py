from abc import abstractmethod, ABC

import einops
import torch

from cryostar.utils.misc import ASSERT_SHAPE
from cryostar.utils.rotation_conversion import axis_angle_to_matrix


class DeformerProtocol(ABC):

    @abstractmethod
    def transform(self, deformation, coords):
        """
        Input:
            deformation: (bsz, _)
            coords: (num_coords, 3)

        Returns:
            (bsz, num_coords, 3)
        """


class E3Deformer(torch.nn.Module, DeformerProtocol):

    def transform(self, deformation, coords):
        ASSERT_SHAPE(coords, (None, 3))
        ASSERT_SHAPE(deformation, (None, coords.shape[0] * 3))

        bsz = deformation.shape[0]
        shift = deformation.reshape(bsz, -1, 3)
        return shift + coords


class NMADeformer(torch.nn.Module, DeformerProtocol):
    def __init__(self, modes: torch.FloatTensor) -> None:
        super().__init__()
        modes = einops.rearrange(
            modes, "(num_coords c3) num_modes -> num_modes num_coords c3", c3=3
        )
        self.register_buffer("modes", modes)
        self.num_modes = modes.shape[0]
        self.num_coords = modes.shape[1]

    def transform(self, deformation, coords):
        ASSERT_SHAPE(coords, (self.num_coords, 3))
        ASSERT_SHAPE(deformation, (None, 6 + self.num_modes))

        axis_angle = deformation[..., :3]
        translation = deformation[..., 3:6] * 10
        nma_coeff = deformation[..., 6:]
        rotation_matrix = axis_angle_to_matrix(axis_angle)

        nma_deform_e3 = einops.einsum(
            nma_coeff, self.modes, "bsz num_modes, num_modes num_coords c3 -> bsz num_coords c3"
        )
        rotated_coords = einops.einsum(rotation_matrix, nma_deform_e3 + coords,
                                       "bsz c31 c32, bsz num_coords c31 -> bsz num_coords c32")
        deformed_coords = rotated_coords + einops.rearrange(translation, "bsz c3 -> bsz 1 c3")
        return deformed_coords
