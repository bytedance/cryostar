import os
from dataclasses import dataclass, field

import mrcfile
import numpy as np
import starfile
import torch
import torch.fft
import torchvision.transforms.functional as tvf
from torch.utils.data import Dataset

from cryostar.utils.fft_utils import primal_to_fourier_2d, fourier_to_primal_2d
from cryostar.utils.geom_utils import euler_angles2matrix


class Mask(torch.nn.Module):

    def __init__(self, im_size, rad):
        super(Mask, self).__init__()

        mask = torch.lt(torch.linspace(-1, 1, im_size)[None]**2 + torch.linspace(-1, 1, im_size)[:, None]**2, rad**2)
        # float for pl ddp broadcast compatible
        self.register_buffer('mask', mask.float())
        self.num_masked = torch.sum(mask).item()

    def forward(self, x):
        return x * self.mask


#yapf: disable
@dataclass
class StarfileDatasetConfig:
    path_to_file: str
    file:         str
    side_len:     int   = field(default=None,
        metadata={"help": "If specifid, the image will be resized."})
    mask_rad:     float = None
    scale_images: float = 1.0
    power_images:  float = field(default=1.0,
        metadata={"help": "Change the power of the signal by mutliplying a constant number."})
    no_trans:     bool  = False
    no_rot:       bool  = False
    invert_hand:  bool  = field(default=False,
        metadata={"help": "Invert handedness when reading relion data."})
#yapf: enable


class StarfileDataSet(Dataset):

    def __init__(self, cfg: StarfileDatasetConfig):
        super(StarfileDataSet, self).__init__()
        self.cfg = cfg
        self.df = starfile.read(os.path.join(cfg.path_to_file, cfg.file))
        if "optics" not in self.df:
            df = self.df
            # yapf: disable
            self.df = {
                "optics": {
                    "rlnVoltage": {0: 300.0},
                    "rlnSphericalAberration": {0: 2.7},
                    "rlnAmplitudeContrast": {0: 0.1},
                    "rlnOpticsGroup": {0: 1},
                    "rlnImageSize": {0: 64},
                    "rlnImagePixelSize": {0: 3.77}
                },
                "particles": df
            }
            # yapf: enable
        self.num_projs = len(self.df["particles"])

        self.true_sidelen = self.df["optics"]["rlnImageSize"][0]
        if cfg.side_len is None:
            self.vol_sidelen = self.df["optics"]["rlnImageSize"][0]
        else:
            self.vol_sidelen = cfg.side_len

        if cfg.mask_rad is not None:
            self.mask = Mask(self.vol_sidelen, cfg.mask_rad)

        self.f_mu = None
        self.f_std = None
        self._apix = None

    @property
    def apix(self):
        return self._apix

    @apix.setter
    def apix(self, apix):
        self._apix = apix

    def __len__(self):
        return self.num_projs

    def estimate_normalization(self):
        if self.f_mu is None and self.f_std is None:
            f_sub_data = []
            for i in range(0, len(self), len(self) // 1000):
                f_sub_data.append(self[i]["fproj"])
            f_sub_data = torch.cat(f_sub_data, dim=0)
            # self.f_mu = torch.mean(f_sub_data)
            self.f_mu = 0.0  # just follow cryodrgn
            self.f_std = torch.std(f_sub_data)
            print(f"Fourier mu/std: {self.f_mu:.5f}/{self.f_std:.5f}")
        else:
            raise Exception("The normalization factor has been estimated!")

    def __getitem__(self, idx):
        particle = self.df["particles"].iloc[idx]
        try:
            # Load particle image from mrcs file
            imgname_raw = particle["rlnImageName"]
            imgnamedf = particle["rlnImageName"].split("@")
            mrc_path = os.path.join(self.cfg.path_to_file, imgnamedf[1])
            pidx = int(imgnamedf[0]) - 1
            with mrcfile.mmap(mrc_path, mode="r", permissive=True) as mrc:
                if mrc.data.ndim > 2:
                    proj = torch.from_numpy(np.array(mrc.data[pidx])).float() * self.cfg.scale_images
                else:
                    # the mrcs file can contain only one particle
                    proj = torch.from_numpy(np.array(mrc.data)).float() * self.cfg.scale_images
            if len(proj.shape) == 2:
                proj = proj[None, :, :]  # add a dummy channel (for consistency w/ img fmt)
            else:
                assert len(proj.shape) == 3 and proj.shape[0] == 1  # some starfile already have a dummy channel

            if self.vol_sidelen != self.true_sidelen:
                proj = tvf.resize(proj, [self.vol_sidelen] * 2, antialias=True)

            if self.cfg.mask_rad is not None:
                proj = self.mask(proj)

        except Exception as e:
            print(f"WARNING: Particle image {imgname_raw} invalid! Setting to zeros.")
            print(e)
            proj = torch.zeros(self.vol_sidelen, self.vol_sidelen)
            proj = proj[None, :, :]

        if self.cfg.power_images != 1.0:
            proj *= self.cfg.power_images

        # Generate CTF from CTF paramaters
        defocusU = torch.from_numpy(np.array(particle["rlnDefocusU"] / 1e4, ndmin=2)).float()
        defocusV = torch.from_numpy(np.array(particle["rlnDefocusV"] / 1e4, ndmin=2)).float()
        angleAstigmatism = torch.from_numpy(np.radians(np.array(particle["rlnDefocusAngle"], ndmin=2))).float()

        # Read "GT" orientations
        if self.cfg.no_rot:
            rotmat = torch.eye(3).float()
        else:
            # yapf: disable
            rotmat = torch.from_numpy(euler_angles2matrix(
                np.radians(-particle["rlnAngleRot"]),
                np.radians(particle["rlnAngleTilt"]) * (-1 if self.cfg.invert_hand else 1),
                np.radians(-particle["rlnAnglePsi"]))
            ).float()
            # yapf: enable

        # Read "GT" shifts
        if self.cfg.no_trans:
            shiftX = torch.tensor([0.])
            shiftY = torch.tensor([0.])
        else:
            # support old star file
            # Particle translations used to be in pixels (rlnOriginX and rlnOriginY) but this changed to Angstroms
            # (rlnOriginXAngstrom and rlnOriginYAngstrom) in relion 3.1.
            # https://relion.readthedocs.io/en/release-3.1/Reference/Conventions.html
            if "rlnOriginXAngst" in particle:
                shiftX = torch.from_numpy(np.array(particle["rlnOriginXAngst"], dtype=np.float32))
                shiftY = torch.from_numpy(np.array(particle["rlnOriginYAngst"], dtype=np.float32))
            else:
                shiftX = torch.from_numpy(np.array(particle["rlnOriginX"] * self._apix, dtype=np.float32))
                shiftY = torch.from_numpy(np.array(particle["rlnOriginY"] * self._apix, dtype=np.float32))

        fproj = primal_to_fourier_2d(proj)

        if self.f_mu is not None:
            fproj = (fproj - self.f_mu) / self.f_std
            proj = fourier_to_primal_2d(fproj).real

        in_dict = {
            "proj": proj,
            "rotmat": rotmat,
            "defocusU": defocusU,
            "defocusV": defocusV,
            "shiftX": shiftX,
            "shiftY": shiftY,
            "angleAstigmatism": angleAstigmatism,
            "idx": torch.tensor(idx, dtype=torch.long),
            "fproj": fproj,
            "imgname_raw": imgname_raw
        }

        if "rlnClassNumber" in particle:
            in_dict["class_id"] = particle["rlnClassNumber"]

        return in_dict
