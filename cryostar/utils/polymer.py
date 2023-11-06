import dataclasses

import gemmi
import numpy as np
import biotite.structure as struc

from cryostar.utils.pdb_tools import bt_read_pdb

# careful about the order
AA_ATOMS = ("CA", )
NT_ATOMS = ("C1'", )


def get_num_electrons(atom_arr):
    return np.sum(np.array([gemmi.Element(x).atomic_number for x in atom_arr.element]))


@dataclasses.dataclass
class Polymer:
    chain_id: np.ndarray
    res_id: np.ndarray
    res_name: np.ndarray
    coord: np.ndarray
    atom_name: np.ndarray
    element: np.ndarray
    num_electron: np.ndarray

    def __init__(self, num):
        self.chain_id = np.empty(num, dtype="U4")
        self.res_id = np.zeros(num, dtype=int)
        self.res_name = np.empty(num, dtype="U3")
        self.coord = np.zeros((num, 3), dtype=np.float32)
        self.atom_name = np.empty(num, dtype="U6")
        self.element = np.empty(num, dtype="U2")
        self.num_electron = np.zeros(num, dtype=int)

    def __setitem__(self, index, kwargs):
        assert set(kwargs.keys()).issubset(f.name for f in dataclasses.fields(self))
        for k, v in kwargs.items():
            getattr(self, k)[index] = v

    def __getitem__(self, index):
        return {f.name: getattr(self, f.name)[index] for f in dataclasses.fields(self)}

    def __len__(self):
        return len(self.chain_id)

    @property
    def num_amino_acids(self):
        return np.sum(np.isin(self.atom_name, AA_ATOMS))

    @property
    def num_nucleotides(self):
        return np.sum(np.isin(self.atom_name, NT_ATOMS))

    @property
    def num_chains(self):
        return len(np.unique(self.chain_id))

    @classmethod
    def from_atom_arr(cls, atom_arr):
        assert isinstance(atom_arr, struc.AtomArray)

        nt_arr = atom_arr[struc.filter_nucleotides(atom_arr)]
        aa_arr = atom_arr[struc.filter_amino_acids(atom_arr)]

        num = 0
        if len(aa_arr) > 0:
            num += struc.get_residue_count(aa_arr)
        if len(nt_arr) > 0:
            for res in struc.residue_iter(nt_arr):
                valid_atoms = set(res.atom_name).intersection(NT_ATOMS)
                if len(valid_atoms) <= 0:
                    raise UserWarning(f"Nucleotides doesn't contain {' or '.join(NT_ATOMS)}.")
                else:
                    num += len(valid_atoms)
        meta = cls(num)

        def _update_res(tmp_res, kind="aa"):
            nonlocal pos

            if kind == "aa":
                using_atom_names = AA_ATOMS
                filtered_res = tmp_res[struc.filter_peptide_backbone(tmp_res)]
            elif kind == "nt":
                using_atom_names = NT_ATOMS
                filtered_res = tmp_res
            else:
                raise NotImplemented

            valid_atom_names = set(tmp_res.atom_name).intersection(using_atom_names)

            for select_atom_name in valid_atom_names:
                meta[pos] = {
                    "chain_id": tmp_res.chain_id[0],
                    "res_id": tmp_res.res_id[0],
                    "res_name": tmp_res.res_name[0],
                    "coord": filtered_res[filtered_res.atom_name == select_atom_name].coord,
                    "atom_name": select_atom_name,
                    "element": filtered_res[filtered_res.atom_name == select_atom_name].element[0],
                    "num_electron": get_num_electrons(tmp_res) // len(valid_atom_names)
                }
                pos += 1

        def _update(tmp_arr, kind="aa"):
            nonlocal pos
            for chain in struc.chain_iter(tmp_arr):
                for tmp_res in struc.residue_iter(chain):
                    _update_res(tmp_res, kind)

        pos = 0

        if len(aa_arr) > 0:
            _update(aa_arr, kind="aa")
        if len(nt_arr) > 0:
            _update(nt_arr, kind="nt")

        assert pos == num
        return meta

    @classmethod
    def from_pdb(cls, file_path):
        atom_arr = bt_read_pdb(file_path)
        if atom_arr.stack_depth() > 1:
            print("PDB file contains more than 1 models, select the 1st model")
        atom_arr = atom_arr[0]
        return Polymer.from_atom_arr(atom_arr)

    def to_atom_arr(self):
        num = len(self)
        atom_arr = struc.AtomArray(num)
        atom_arr.coord = self.coord

        for f in dataclasses.fields(self):
            if f.name != "coord" and f.name in atom_arr.get_annotation_categories():
                atom_arr.set_annotation(f.name, getattr(self, f.name))
        # atom_arr.atom_name[atom_arr.atom_name == "R"] = "CB"
        return atom_arr
