# This file contains code modified from parse_ctf_star.py and parse_ctf_csparc.py 
# available at https://github.com/ml-struct-bio/cryodrgn/tree/main/cryodrgn/commands
# 
# Modifications include the consolidation of 'ctf' parameter name and 
# accompanying modifications.
# 
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

from typing import Optional

import numpy as np
import starfile

import torch

CTF_FIELDS = {
    "side_shape": "Image size",
    "apix": "Pixel size (A)",
    "dfu": "DefocusU (A)",
    "dfv": "DefocusV (A)",
    "dfa": "DefocusAngle (deg)",
    "volt": "Voltage (kV)",
    "cs": "SphericalAberration (mm)",
    "w": "AmplitudeContrast",
    "ps": "Phase shift (deg)"
}


# kwargs contain keys as CTF_FIELDS.keys() to override some values
def parse_ctf_star(f_path, **kwargs):
    """
    Parse CTF information from RELION .star file, return a (N, 9) array

    Args:
        f_path: starfile path
        **kwargs:

    Returns:
        ctf_params (N, 9) numpy array
    """
    df = starfile.read(f_path)

    overrides = {}

    try:
        side_shape = int(df["optics"].loc[0, "rlnImageSize"])
        apix = df["optics"].loc[0, "rlnImagePixelSize"]
    except Exception:
        assert "side_shape" in kwargs and "apix" in kwargs, "side_shape, apix must be provided."
        side_shape = kwargs["side_shape"]
        apix = kwargs["apix"]

    if "optics" in df:
        assert len(df["optics"]) == 1, "Currently only support one optics group."
        overrides["rlnVoltage"] = df["optics"].loc[0, "rlnVoltage"]
        overrides["rlnSphericalAberration"] = df["optics"].loc[0, "rlnSphericalAberration"]
        overrides["rlnAmplitudeContrast"] = df["optics"].loc[0, "rlnAmplitudeContrast"]

    if "particles" in df:
        df = df["particles"]

    if "volt" in kwargs:
        overrides["rlnVoltage"] = float(kwargs.get("volt"))
    if "cs" in kwargs:
        overrides["rlnSphericalAberration"] = float(kwargs.get("cs"))
    if "w" in kwargs:
        overrides["rlnAmplitudeContrast"] = float(kwargs.get("w"))
    if "ps" in kwargs:
        overrides["rlnPhaseShift"] = float(kwargs.get("ps"))

    num = len(df)
    ctf_params = np.zeros((num, 9))
    ctf_params[:, 0] = side_shape
    ctf_params[:, 1] = apix
    for i, header in enumerate([
            "rlnDefocusU",
            "rlnDefocusV",
            "rlnDefocusAngle",
            "rlnVoltage",
            "rlnSphericalAberration",
            "rlnAmplitudeContrast",
            "rlnPhaseShift",
    ]):
        if header in overrides:
            ctf_params[:, i + 2] = overrides[header]
        else:
            ctf_params[:, i + 2] = df[header].values if header in df else None
    return ctf_params


def parse_ctf_cs(f_path, **kwargs):
    """
    Parse CTF information from CryoSPARC .cs file, return a (N, 9) array

    Args:
        f_path: .cs file path
        **kwargs:

    Returns:
        ctf_params (N, 9) numpy array
    """
    rec_arr = np.load(f_path)

    try:
        side_shape = rec_arr["blob/shape"][0][0]
        apix = rec_arr["blob/psize_A"]
    except Exception:
        assert "side_shape" in kwargs and "apix" in kwargs, "side_shape, apix must be provided."
        side_shape = kwargs["side_shape"]
        apix = kwargs["apix"]

    num = len(rec_arr)
    ctf_params = np.zeros((num, 9))
    ctf_params[:, 0] = side_shape
    ctf_params[:, 1] = apix
    fields = (
        "ctf/df1_A",
        "ctf/df2_A",
        "ctf/df_angle_rad",
        "ctf/accel_kv",
        "ctf/cs_mm",
        "ctf/amp_contrast",
        "ctf/phase_shift_rad",
    )
    for i, f in enumerate(fields):
        ctf_params[:, i + 2] = rec_arr[f]
        if f in ("ctf/df_angle_rad", "ctf/phase_shift_rad"):  # convert to degrees
            ctf_params[:, i + 2] *= 180 / np.pi
    return ctf_params


def print_ctf_params(params):
    """
    Print one ctf params

    Args:
        params: numpy array with shape (9, )
    """
    list(CTF_FIELDS.values())
    assert len(params) == 9
    for k, v in zip(CTF_FIELDS.values(), params):
        print(f"{k:30}: {v}")


def compute_ctf(
    freqs: torch.Tensor,
    dfu: torch.Tensor,
    dfv: torch.Tensor,
    dfang: torch.Tensor,
    volt: torch.Tensor,
    cs: torch.Tensor,
    w: torch.Tensor,
    phase_shift: Optional[torch.Tensor] = None,
    bfactor: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Compute the 2D CTF

    Input:
        freqs (np.ndarray) Nx2 or BxNx2 tensor of 2D spatial frequencies
        dfu (float or Bx1 tensor): DefocusU (Angstrom)
        dfv (float or Bx1 tensor): DefocusV (Angstrom)
        dfang (float or Bx1 tensor): DefocusAngle (degrees)
        volt (float or Bx1 tensor): accelerating voltage (kV)
        cs (float or Bx1 tensor): spherical aberration (mm)
        w (float or Bx1 tensor): amplitude contrast ratio
        phase_shift (float or Bx1 tensor): degrees
        bfactor (float or Bx1 tensor): envelope fcn B-factor (Angstrom^2)
    """
    # convert units
    volt = volt * 1000
    cs = cs * 10**7
    dfang = dfang * np.pi / 180
    if phase_shift is None:
        phase_shift = torch.tensor(0)
    phase_shift = phase_shift * np.pi / 180

    # lam = sqrt(h^2/(2*m*e*Vr)); Vr = V + (e/(2*m*c^2))*V^2
    lam = 12.2639 / torch.sqrt(volt + 0.97845e-6 * volt**2)
    x = freqs[..., 0]
    y = freqs[..., 1]
    ang = torch.arctan2(y, x)
    s2 = x**2 + y**2
    df = 0.5 * (dfu + dfv + (dfu - dfv) * torch.cos(2 * (ang - dfang)))
    gamma = (2 * torch.pi * (-0.5 * df * lam * s2 + 0.25 * cs * lam**3 * s2**2) - phase_shift)
    ctf = torch.sqrt(1 - w**2) * torch.sin(gamma) - w * torch.cos(gamma)
    if bfactor is not None:
        ctf *= torch.exp(-bfactor / 4 * s2)
    return ctf
