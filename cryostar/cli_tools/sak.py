import sys
import os.path as osp
from pathlib import Path

import mrcfile
import numpy as np

import torch

from cryostar.utils.pdb_tools import bt_read_pdb, bt_save_pdb
from cryostar.utils.mrc_tools import save_mrc
from cryostar.utils.polymer import Polymer
from cryostar.gmm.gmm import EMAN2Grid, Gaussian, canonical_density


def _check_valid_file_path(file_path):
    if not (Path(file_path).is_file() and Path(file_path).exists()):
        print(f"{file_path} is not a valid file path.")
        sys.exit(1)


def _get_file_name(file_path):
    return osp.splitext(osp.basename(file_path))[0]


def show_mrc_info():
    """
    Show meta info of an .mrc file.

    Usage:
    show_mrc_info <mrc_file_path.mrc>
    """
    if len(sys.argv) != 2:
        print("In order to view information from your .mrc file, please use the correct command format "
              "as:\nshow_mrc_info <mrc_file_path.mrc>")
        sys.exit(1)
    if sys.argv[1] in ("-h", "--help"):
        print(show_mrc_info.__doc__)
        return

    mrc_file_path = str(sys.argv[1])
    _check_valid_file_path(mrc_file_path)

    with mrcfile.open(mrc_file_path) as m:
        print(f"The mrcfile contains volume data with:\n"
              f"shape (nz, ny, nx):     {m.data.shape}\n"
              f"voxel_size/A (x, y, z): {m.voxel_size}\n"
              f"origin/A (x, y, z):     {m.header.origin}")


def center_origin():
    """
    Centers the origin of PDB and MRC file

    This function moves the origin of coordinates for both PDB and MRC files to the
    center of the MRC three-dimensional data matrix, so that the center of the 3D
    data matrix becomes (0,0,0). It then saves the adjusted files in the current
    directory with a '_centered' suffix.

    Usage:
    center_origin <reference_structure_path.pdb> <consensus_map_path.mrc>

    Args:
    reference_structure_path (str): The path to the input PDB file.
    consensus_map_path (str): The path to the input MRC file.
    """
    if len(sys.argv) != 3:
        if len(sys.argv) == 2 and sys.argv[1] in ("-h", "--help"):
            print(center_origin.__doc__)
            return
        else:
            print("please use the correct command format as:\n"
                  "center_origin <reference_structure_path.pdb> <consensus_map_path.mrc>")
            sys.exit(1)

    pdb_file_path = str(sys.argv[1])
    mrc_file_path = str(sys.argv[2])

    _check_valid_file_path(pdb_file_path)
    _check_valid_file_path(mrc_file_path)

    with mrcfile.open(mrc_file_path) as m:
        if m.voxel_size.x == m.voxel_size.y == m.voxel_size.z and np.all(np.asarray(m.data.shape) == m.data.shape[0]):
            new_origin = (- m.data.shape[0] // 2 * m.voxel_size.x, ) * 3
        else:
            print("The voxel sizes or shapes differ across the three axes in the three-dimensional data.")
            new_origin = (- m.data.shape[2] // 2 * m.voxel_size.x, - m.data.shape[1] // 2 * m.voxel_size.y,
                          - m.data.shape[0] // 2 * m.voxel_size.z)
        save_mrc(m.data.copy(), _get_file_name(mrc_file_path) + "_centered.mrc",
                 m.voxel_size, new_origin)
        print(f"Result centered MRC saved to {_get_file_name(mrc_file_path)}_centered.mrc.")

        atom_arr = bt_read_pdb(pdb_file_path)[0]
        atom_arr.coord += np.asarray(new_origin)
        bt_save_pdb(_get_file_name(pdb_file_path) + "_centered.pdb", atom_arr)
        print(f"Result centered PDB saved to {_get_file_name(pdb_file_path)}_centered.pdb.")


def generate_gaussian_density():
    """
    Generate Gaussian density corresponding to a given PDB file

    Note that the input PDB file must be centered before.

    Usages:
    generate_gaussian_density <pdb_file_path.pdb> <shape> <apix>
    generate_gaussian_density <pdb_file_path.pdb> <shape> <apix> [<save_path.mrc>]

    Args:
    pdb_file_path (str): The path to the input PDB file.
    shape (int): An integer that represents the shape of the Gaussian density.
    apix (float): A floating-point value that reflects the pixel size in Angstrom.
    save_path (str, optional): The path to save the resultant Gaussian density. If not provided,
                               the function will store the data in the current working directory.
    """
    if len(sys.argv) not in (4, 5):
        if len(sys.argv) == 2 and sys.argv[1] in ("-h", "--help"):
            print(generate_gaussian_density.__doc__)
            return
        else:
            print("please use the correct command format as:\n"
                  "generate_gaussian_density <pdb_file_path.pdb> <shape> <apix>\n"
                  "or generate_gaussian_density <pdb_file_path.pdb> <shape> <apix> [<save_path.mrc>]")
            sys.exit(1)

    # input params
    pdb_file_path = str(sys.argv[1])
    shape = int(sys.argv[2])
    apix = float(sys.argv[3])
    if len(sys.argv) == 5:
        save_path = str(sys.argv[4])
    else:
        save_path = _get_file_name(pdb_file_path) + "_gaussian.mrc"

    _check_valid_file_path(pdb_file_path)

    #
    atom_arr = bt_read_pdb(pdb_file_path)[0]
    meta = Polymer.from_atom_arr(atom_arr)

    ref_centers = torch.from_numpy(meta.coord).float()
    ref_amps = torch.from_numpy(meta.num_electron).float()
    ref_sigmas = torch.ones_like(ref_amps)
    ref_sigmas.fill_(2.)

    grid = EMAN2Grid(side_shape=shape, voxel_size=apix)

    vol = canonical_density(
        gauss=Gaussian(
            mus=ref_centers,
            sigmas=ref_sigmas,
            amplitudes=ref_amps),
        line_grid=grid.line())
    vol = vol.permute(2, 1, 0).cpu().numpy()

    save_mrc(vol, save_path, apix, -shape // 2 * apix)
    print(f"Result Gaussian density saved to {save_path}.")
