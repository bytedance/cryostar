import os.path as osp
from pathlib import Path
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
    dataset_dir:     str
    starfile_path:   str
    # if is not specified, the following apix, and side_shape will be inferred from starfile
    apix:            float = None
    side_shape:      int   = None
    # down-sample the original image or not
    down_side_shape: int   = None
    # apply a circular mask on input image or not
    mask_rad:        float = None
    # change image values
    scale_images:    float = 1.0
    power_images:    float = field(
        default=1.0,
        metadata={"help": "Change the power of the signal by multiplying a constant number."})
    # ignore pose from starfile or not
    ignore_trans:    bool  = False
    ignore_rots:     bool  = False
    # invert_hand:     bool  = field(
    #     default=False,
    #     metadata={"help": "Invert handedness when reading relion data."})
#yapf: enable


class StarfileDataSet(Dataset):

    def __init__(self, cfg: StarfileDatasetConfig):
        super().__init__()
        self.cfg = cfg
        self.df = starfile.read(Path(cfg.starfile_path))

        if "optics" in self.df:
            optics_df = self.df["optics"]
            particles_df = self.df["particles"]
        else:
            optics_df = None
            particles_df = self.df
        self.particles_df = particles_df

        if cfg.apix is None:
            if optics_df is not None and "rlnImagePixelSize" in optics_df:
                self.apix = float(optics_df["rlnImagePixelSize"][0])
                print(f"Infer dataset apix={self.apix} from first optic group.")
            elif "rlnDetectorPixelSize" in particles_df and "rlnMagnification" in particles_df:
                self.apix = float(particles_df["rlnDetectorPixelSize"][0] / particles_df["rlnMagnification"][0] * 1e4)
                print(f"Infer dataset apix={self.apix} from first particle meta data.")
            else:
                raise AttributeError("Cannot parse apix from starfile, please set it in config by hand.")
        else:
            self.apix = cfg.apix

        if cfg.side_shape is None:
            tmp_mrc_path = osp.join(cfg.dataset_dir, particles_df["rlnImageName"][0].split('@')[-1])
            with mrcfile.mmap(tmp_mrc_path, mode="r", permissive=True) as m:
                self.side_shape = m.data.shape[-1]
            print(f"Infer dataset side_shape={self.side_shape} from the 1st particle.")
        else:
            self.side_shape = cfg.side_shape

        self.num_proj = len(particles_df)

        self.down_side_shape = self.side_shape
        if cfg.down_side_shape is not None:
            self.down_side_shape = cfg.down_side_shape

        if cfg.mask_rad is not None:
            self.mask = Mask(self.down_side_shape, cfg.mask_rad)

        self.f_mu = None
        self.f_std = None

    def __len__(self):
        return self.num_proj

    def estimate_normalization(self):
        if self.f_mu is None and self.f_std is None:
            f_sub_data = []
            # I have checked that the standard deviation of 10/100/1000 particles is similar
            for i in range(0, len(self), len(self) // 100):
                f_sub_data.append(self[i]["fproj"])
            f_sub_data = torch.cat(f_sub_data, dim=0)
            # self.f_mu = torch.mean(f_sub_data)
            self.f_mu = 0.0  # just follow cryodrgn
            self.f_std = torch.std(f_sub_data).item()
        else:
            raise Exception("The normalization factor has been estimated!")

    def __getitem__(self, idx):
        item_row = self.particles_df.iloc[idx]
        try:
            img_name_raw = item_row["rlnImageName"]
            in_mrc_idx, img_name = item_row["rlnImageName"].split("@")
            in_mrc_idx = int(in_mrc_idx) - 1
            mrc_path = osp.join(self.cfg.dataset_dir, img_name)
            with mrcfile.mmap(mrc_path, mode="r", permissive=True) as mrc:
                if mrc.data.ndim > 2:
                    proj = torch.from_numpy(np.array(mrc.data[in_mrc_idx])).float() * self.cfg.scale_images
                else:
                    # the mrcs file can contain only one particle
                    proj = torch.from_numpy(np.array(mrc.data)).float() * self.cfg.scale_images

            # get (1, side_shape, side_shape) proj
            if len(proj.shape) == 2:
                proj = proj[None, :, :]  # add a dummy channel (for consistency w/ img fmt)
            else:
                assert len(proj.shape) == 3 and proj.shape[0] == 1  # some starfile already have a dummy channel

            # down-sample
            if self.down_side_shape != self.side_shape:
                proj = tvf.resize(proj, [self.down_side_shape, ] * 2, antialias=True)

            if self.cfg.mask_rad is not None:
                proj = self.mask(proj)

        except Exception as e:
            print(f"WARNING: Particle image {img_name_raw} invalid! Setting to zeros.")
            print(e)
            proj = torch.zeros(1, self.down_side_shape, self.down_side_shape)

        if self.cfg.power_images != 1.0:
            proj *= self.cfg.power_images

        # Generate CTF from CTF paramaters
        defocusU = torch.from_numpy(np.array(item_row["rlnDefocusU"] / 1e4, ndmin=2)).float()
        defocusV = torch.from_numpy(np.array(item_row["rlnDefocusV"] / 1e4, ndmin=2)).float()
        angleAstigmatism = torch.from_numpy(np.radians(np.array(item_row["rlnDefocusAngle"], ndmin=2))).float()

        # Read "GT" orientations
        if self.cfg.ignore_rots:
            rotmat = torch.eye(3).float()
        else:
            # yapf: disable
            rotmat = torch.from_numpy(euler_angles2matrix(
                np.radians(-item_row["rlnAngleRot"]),
                # np.radians(particle["rlnAngleTilt"]) * (-1 if self.cfg.invert_hand else 1),
                np.radians(-item_row["rlnAngleTilt"]),
                np.radians(-item_row["rlnAnglePsi"]))
            ).float()
            # yapf: enable

        # Read "GT" shifts
        if self.cfg.ignore_trans:
            shiftX = torch.tensor([0.])
            shiftY = torch.tensor([0.])
        else:
            # support early starfile formats
            # Particle translations used to be in pixels (rlnOriginX and rlnOriginY) but this changed to Angstroms
            # (rlnOriginXAngstrom and rlnOriginYAngstrom) in relion 3.1.
            # https://relion.readthedocs.io/en/release-3.1/Reference/Conventions.html
            if "rlnOriginXAngst" in item_row:
                shiftX = torch.from_numpy(np.array(item_row["rlnOriginXAngst"], dtype=np.float32))
                shiftY = torch.from_numpy(np.array(item_row["rlnOriginYAngst"], dtype=np.float32))
            else:
                shiftX = torch.from_numpy(np.array(item_row["rlnOriginX"] * self.apix, dtype=np.float32))
                shiftY = torch.from_numpy(np.array(item_row["rlnOriginY"] * self.apix, dtype=np.float32))

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
            "imgname_raw": img_name_raw
        }

        if "rlnClassNumber" in item_row:
            in_dict["class_id"] = item_row["rlnClassNumber"]

        return in_dict
