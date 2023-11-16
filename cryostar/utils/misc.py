import argparse
import functools
import os
import os.path as osp
import random
import time
from shutil import copyfile
from typing import Union

import biotite.structure as struc
import biotite.structure.io as strucio
import numpy as np
import torch
import einops
from Bio.PDB import PDBIO, PDBParser
from Bio.PDB.StructureBuilder import StructureBuilder
from biotite.structure import Atom, array, stack
from biotite.structure.io.pdb import PDBFile
from mmengine import Config, MMLogger, print_log
from mmengine.config import DictAction
from mmengine.utils import mkdir_or_exist
from torch.utils.tensorboard import SummaryWriter

log_to_current = functools.partial(print_log, logger="current")


def set_seed(seed: int = 42):
    """Sets random sets for torch operations.

    Parameters
    ----------
    seed (int, optional): Random seed to set. Defaults to 42.

    Returns
    -------

    """
    random.seed(seed)
    np.random.seed(seed)
    # Set the seed for general torch operations
    torch.manual_seed(seed)
    # Set the seed for CUDA torch operations (ones that happen on the GPU)
    torch.cuda.manual_seed_all(seed)


def chain(arg, *funcs):
    result = arg
    for f in funcs:
        result = f(result)
    return result


def convert_to_numpy(*args):
    ret = []
    for arg in args:
        if isinstance(arg, torch.Tensor):
            ret.append(arg.detach().cpu().numpy())
        else:
            ret.append(arg)
    if len(args) == 1:
        return ret[0]
    else:
        return ret


def CHECK_SHAPE(tensor, expected_shape):
    if len(tensor.shape) != len(expected_shape):
        return False
    for a, b in zip(tuple(tensor.shape), tuple(expected_shape)):
        if b is not None and a != b:
            return False
    return True


def ASSERT_SHAPE(tensor, expected_shape):
    assert CHECK_SHAPE(tensor, expected_shape), f"Expected shape {expected_shape}, got {tensor.shape}"


def parse_mmengine_args(override_mode="default"):
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('config', help='config file path')
    args = None

    if override_mode == "default":
        """
        Use case:
            args = parser.parse_args(["config", "--cfg-options", "data=1", "flag=false"])
            -> Namespace(cfg_options={'data': 1, 'flag': False}, config='config')
        """
        parser.add_argument("--cfg-options", nargs='+', action=DictAction)
        args = parser.parse_args()
    elif override_mode == "key-value":
        """
        Use case:
            args, unknown = parser.parse_known_args(["config", "--data", "1", "--flag", "false"])
            ...
            -> Namespace(cfg_options={'data': 1, 'flag': False}, config='config')
        """
        args, unknown = parser.parse_known_args()
        assert len(unknown) % 2 == 0
        cfg_options = {}
        while len(unknown) != 0:
            k = unknown.pop(0)
            v = unknown.pop(0)
            assert k.startswith("--") and not v.startswith("--")
            cfg_options[k[2:]] = DictAction._parse_iterable(v)
        args.cfg_options = cfg_options

    return args


def flatten_nested_dict(nested: Union[dict, Config]) -> dict:
    """
        Inputs: 
            {"a": {"b": 1}, "c": 2} 
        Returns:
            {"a.b": 1, "c": 2}
    """
    stack = list(nested.items())
    ans = {}
    while stack:
        key, val = stack.pop()
        if isinstance(val, dict):
            for sub_key, sub_val in val.items():
                stack.append((f"{key}.{sub_key}", sub_val))
        else:
            ans[key] = val
    return ans


def warmup(warmup_step, lower=0.0, upper=1.0):
    value_range = (upper - lower)

    def run(cur_step):
        if cur_step < warmup_step:
            return (cur_step / warmup_step) * value_range
        else:
            return upper

    return run


def init_mmengine_config(args):
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        flatten_cfg = flatten_nested_dict(cfg)
        for ele in args.cfg_options.keys():
            if ele not in flatten_cfg:
                raise Exception(f"{ele} not in config!")
        cfg.merge_from_dict(args.cfg_options)
    return cfg


def init_mmengine_exp(args,
                      exp_prefix='',
                      backup_list=None,
                      inplace=True,
                      work_dir_name="work_dirs",
                      project_name="cryostar",
                      tensorboard=False):
    """
        Initialization of an experiment, including creating the experiment's folder, 
        backuping some scripts, and so on. The type of `args` is `argparse.Namespace`, 
        where `args.config` points to a config file (*.py).
        
        How to determine an experiment's name? 
        - If `exp_name` is specified, an experiment folder `{work_dir_name}/{exp_name}` will
            be created.
        - Otherwise it will be deduced from `args.config` and `exp_prefix`. If `args.config` 
            is `a/b/c.py`, `exp_prefix` is `foo`, the name is `{work_dir_name}/foo_c`.
    """
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())

    if len(exp_prefix) and not exp_prefix.endswith('_'):
        exp_prefix += '_'

    cfg = init_mmengine_config(args)

    if not hasattr(cfg, "exp_name") or cfg.exp_name == "":
        exp_name = exp_prefix + osp.splitext(osp.basename(args.config))[0]
    else:
        exp_name = cfg.exp_name
    cfg.work_dir = osp.join(work_dir_name, exp_name)

    if not inplace:
        if osp.exists(work_dir_name):
            dir_list = [d for d in os.listdir(work_dir_name) if d.startswith(exp_name)]

            if len(dir_list) > 0:
                new_id = 0
                exist_ids = [
                    int(i.split('_')[-1]) for i in dir_list if len(i.split('_')) > 0 and i.split('_')[-1].isdigit()
                ]
                if len(exist_ids) > 0:
                    new_id = max(exist_ids) + 1
                cfg.work_dir += f"_{new_id:02d}"

    # create work_dir
    mkdir_or_exist(osp.abspath(cfg.work_dir))
    # backup scripts
    if backup_list is not None:
        for p in backup_list:
            copyfile(p, osp.join(cfg.work_dir, osp.basename(p)))
    # dump config
    cfg.dump(osp.join(cfg.work_dir, "config.py"))
    # init the logger before other steps
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    # logger = get_logger(project_name, log_file=log_file)
    logger = MMLogger.get_instance(name=project_name, logger_name=project_name, log_file=log_file)
    if tensorboard:
        writer = SummaryWriter(log_dir=cfg.work_dir, flush_secs=120)
    else:
        writer = None
    logger.info("Config:\n" + cfg.pretty_text)
    return cfg, logger, writer


def _get_next_version(root_dir, dir_name_prefix):
    dir_list = [d for d in os.listdir(root_dir) if d.startswith(dir_name_prefix)]

    new_id = 0

    if len(dir_list) > 0:
        exist_ids = [int(i.split('_')[-1]) for i in dir_list if len(i.split('_')) > 0 and i.split('_')[-1].isdigit()]
        if len(exist_ids) > 0:
            new_id = max(exist_ids) + 1
    return new_id


def pl_init_exp(override_mode="default",
                exp_prefix='',
                backup_list=None,
                inplace=False,
                work_dir_name="work_dirs",
                project_name="cryostar"):
    args = parse_mmengine_args(override_mode=override_mode)
    cfg = init_mmengine_config(args)

    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())

    if len(exp_prefix) and not exp_prefix.endswith('_'):
        exp_prefix += '_'

    if not hasattr(cfg, "exp_name") or cfg.exp_name == "":
        exp_name = exp_prefix + osp.splitext(osp.basename(args.config))[0]
    else:
        exp_name = cfg.exp_name
    cfg.work_dir = osp.join(work_dir_name, exp_name)

    # lightning.fabric.utilities.distributed._distributed_available is True until fit start
    if "WORLD_SIZE" in os.environ:
        _ = MMLogger.get_instance(name="MMLogger", logger_name=project_name + "_fork")
        return cfg

    if not inplace:
        if osp.exists(work_dir_name):
            version = _get_next_version(work_dir_name, exp_name)
            cfg.work_dir += f"_{version:d}"

    # create work_dir
    mkdir_or_exist(osp.abspath(cfg.work_dir))
    # backup scripts
    if backup_list is not None:
        for p in backup_list:
            copyfile(p, osp.join(cfg.work_dir, osp.basename(p)))
    # dump config
    cfg.dump(osp.join(cfg.work_dir, "config.py"))
    # init the logger before other steps
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    # logger = get_logger(project_name, log_file=log_file)
    logger = MMLogger.get_instance(name="MMLogger", logger_name=project_name, log_file=log_file)
    logger.info("Config:\n" + cfg.pretty_text)
    return cfg


def save_pdb(CAs, path, ref_pdb_path):
    parser = PDBParser(PERMISSIVE=True, QUIET=True)
    structure = parser.get_structure(id="temp", file=ref_pdb_path)
    chain = next(structure.get_chains())

    for i, residue in enumerate(chain):
        to_pop = []
        for atom in residue:
            if atom.id != "CA":
                to_pop.append(atom.id)
            else:
                atom.coord = CAs[i]
        for ele in to_pop:
            residue.detach_child(ele)

    io = PDBIO()
    io.set_structure(structure)
    io.save(path)


def load_CAs_from_pdb(file):
    coords = load_NCaC_from_pdb(file)
    coords = coords.reshape(-1, 3, 3)[:, 1, :]
    return coords


def load_NCaC_from_pdb(file):
    source = PDBFile.read(file)
    source_struct = source.get_structure()[0]
    backbone_atoms = source_struct[struc.filter_backbone(source_struct)]
    coords = struc.coord(backbone_atoms)
    return coords


def load_chain_A(pdb_path):
    structure = strucio.load_structure(pdb_path)
    protein_chain = structure[struc.filter_amino_acids(structure) & (structure.chain_id == structure.chain_id[0])]
    return protein_chain


def points_to_pdb(path_to_save, points: np.ndarray):
    struct = StructureBuilder()
    struct.init_structure("1")
    struct.init_seg("1")
    struct.init_model("1")
    struct.init_chain("1")
    for i, point in enumerate(points):
        struct.set_line_counter(i)
        struct.init_residue(f"ALA", " ", i, " ")
        struct.init_atom("CA", point, 0, 1, " ", "CA", "C", element="C")
    struct = struct.get_structure()
    io = PDBIO()
    io.set_structure(struct)
    io.save(path_to_save)


def point_stack_to_pdb(path_to_save, point_stack: np.ndarray):
    """
        Save a dynamic pdb with many models.
        
        Input:
            point_stack:  (num_models, num_atoms, 3)
    """
    assert len(point_stack.shape) == 3
    stack_size = len(point_stack)
    atom_num = len(point_stack[0])
    array_stacks = []
    for stack_idx in range(stack_size):
        array_stacks.append(
            array([
                Atom(point_stack[stack_idx][atom_idx],
                     atom_name="CA",
                     chain_id="A",
                     res_id=atom_idx + 1,
                     hetero=False,
                     element="C",
                     res_name="ALA") for atom_idx in range(atom_num)
            ]))
    array_stacks = stack(array_stacks)
    strucio.save_structure(path_to_save, array_stacks)


def find_rigid_alignment(A, B):
    """
    Code from https://gist.github.com/bougui505/e392a371f5bab095a3673ea6f4976cc8
    
    See: https://en.wikipedia.org/wiki/Kabsch_algorithm
    2-D or 3-D registration with known correspondences.
    Registration occurs in the zero centered coordinate system, and then
    must be transported back.
        Args:
        -    A: Torch tensor of shape (N,D) -- Point Cloud to Align (source)
        -    B: Torch tensor of shape (N,D) -- Reference Point Cloud (target)
        Returns:
        -    R: optimal rotation
        -    t: optimal translation
    Test on rotation + translation and on rotation + translation + reflection
    
        >>> A = torch.tensor([[1., 1.], [2., 2.], [1.5, 3.]], dtype=torch.float)
        >>> R0 = torch.tensor([[np.cos(60), -np.sin(60)], [np.sin(60), np.cos(60)]], dtype=torch.float)
        >>> B = (R0.mm(A.T)).T
        >>> t0 = torch.tensor([3., 3.])
        >>> B += t0
        >>> R, t = find_rigid_alignment(A, B)
        >>> A_aligned = (R.mm(A.T)).T + t
        >>> rmsd = torch.sqrt(((A_aligned - B)**2).sum(axis=1).mean())
        >>> rmsd
        tensor(3.7064e-07)
        >>> B *= torch.tensor([-1., 1.])
        >>> R, t = find_rigid_alignment(A, B)
        >>> A_aligned = (R.mm(A.T)).T + t
        >>> rmsd = torch.sqrt(((A_aligned - B)**2).sum(axis=1).mean())
        >>> rmsd
        tensor(3.7064e-07)
    """
    a_mean = A.mean(axis=0)
    b_mean = B.mean(axis=0)
    A_c = A - a_mean
    B_c = B - b_mean
    # Covariance matrix
    H = A_c.T.mm(B_c)
    U, S, V = torch.svd(H)
    # Rotation matrix
    R = V.mm(U.T)
    # Translation vector
    t = b_mean[None, :] - R.mm(a_mean[None, :].T).T
    t = t.T
    return R, t.squeeze()


def batch_find_rigid_alignment(A, B):
    """
        Input:
            A:  (B, 3)
            B:  (B, 3)
        Returns:
            Rs: (B, 3, 3)
            Ts: (B, 3)
    """
    assert len(A.shape) == 3
    a_mean = A.mean(axis=1, keepdim=True)
    b_mean = B.mean(axis=1, keepdim=True)
    A_c = A - a_mean
    B_c = B - b_mean
    # Covariance matrix
    H = einops.einsum(A_c, B_c, "b n c1, b n c2 -> b c1 c2")

    # The returned V in torch.linalg.svd is different from torch.svd
    #   tensor.mT means tensor.transpose(-1, -2)
    #   tensor.T means tensor.transpose(-1, -2, -3, ...)
    U, S, VmT = torch.linalg.svd(H)
    V = VmT.mT
    # Rotation matrix
    R = einops.einsum(V, U.transpose(2, 1), "b c1 c2, b c2 c3 -> b c1 c3")
    # Translation vector
    t = b_mean - einops.einsum(R, a_mean, "b c1 c2, b n c2 -> b n c1")
    return R, t.squeeze()  


def pretty_dict(x, precision=3):
    """
        Input: 
            x:  a dict (containing torch.Tensor, np.ndarray, float, etc.) to show in a line
            precision:  the precision of float number

        Returns:
            a formatted string, following `fairseq`'s style

        Example:
        ```
            pretty_dict({  
                k1: 1, 
                k2: 1.414213562,
                k3: torch.Tensor(3.1415926535)
            }, 3)

            Output:
            k1 1 | k2 1.414 | k3 3.141
        ```
            
    """
    ret = None
    if isinstance(x, dict):
        ret = []
        for k, v in x.items():
            if isinstance(v, torch.Tensor) or isinstance(v, np.ndarray):
                v = v.item()
            else:
                v = v
            if isinstance(v, float):
                # ret.append("{}: {:.3f}".format(k, v))
                float_formatter = "{}: {:.<PCS>f}".replace("<PCS>", str(precision))
                ret.append(float_formatter.format(k, v))
            else:
                ret.append("{}: {}".format(k, v))
        ret = " | ".join(ret)
    return ret


def create_sphere_mask(d, h, w, center=None, radius=None) -> np.ndarray:
    if center is None:
        center = (int(d / 2), int(h / 2), int(w / 2))
    if radius is None:
        radius = min(center[0], center[1], w - center[0], h - center[1])

    z, y, x = np.ogrid[:d, :h, :w]
    dist_from_center = np.sqrt((x - center[0])**2 + (y - center[1])**2 + (z - center[2])**2)

    mask = dist_from_center <= radius
    return mask


def create_circular_mask(h, w, center=None, radius=None) -> np.ndarray:
    if center is None:  # use the middle of the image
        center = (int(w / 2), int(h / 2))
    if radius is None:  # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w - center[0], h - center[1])

    y, x = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((x - center[0])**2 + (y - center[1])**2)

    mask = dist_from_center <= radius
    return mask

