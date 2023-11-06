import math

import gemmi
import numpy as np

import cryostar.common.residue_constants as rc

e2_atom_electron = {
    'H': (1.0, 1.00794),
    'HO': (1.0, 1.00794),
    'C': (6.0, 12.0107),
    'A': (7.0, 14.00674),
    'N': (7.0, 14.00674),
    'O': (8.0, 15.9994),
    'P': (15.0, 30.973761),
    'K': (19.0, 39.0983),
    'S': (16.0, 32.066),
    'W': (18.0, 1.00794 * 2.0 + 15.9994),
    'AU': (79.0, 196.96655)
}

restype_atom37_electrons = np.zeros([21, 37], dtype=np.float32)
restype_atom14_electrons = np.zeros([21, 14], dtype=np.float32)


def _make_e2_constants():
    for restype, restype_letter in enumerate(rc.restypes):
        resname = rc.restype_1to3[restype_letter]
        restype_name = rc.restype_1to3[restype_letter]
        atom_names = rc.residue_atoms[restype_name]
        for atom_name in atom_names:
            atom_type = rc.atom_order[atom_name]
            # here set the element name to the first character of atom_name, only support normal ATOM, not HETATM
            ele_name = atom_name[0]
            atomic_number = gemmi.Element(ele_name).atomic_number
            restype_atom37_electrons[restype, atom_type] = atomic_number

            atom14idx = rc.restype_name_to_atom14_names[resname].index(atom_name)
            restype_atom14_electrons[restype, atom14idx] = atomic_number


_make_e2_constants()


def restype_atom14_sigmas(resolution=2.8):
    return rc.restype_atom14_mask * resolution / (math.pi * math.sqrt(2))


def restype_atom37_sigmas(resolution=2.8):
    return rc.restype_atom37_mask * resolution / (math.pi * math.sqrt(2))
