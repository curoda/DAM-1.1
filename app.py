import math
from dataclasses import dataclass
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st


# =============================
# Data models
# =============================

@dataclass
class Phase2Config:
    N: int                  # number of geometry points
    W: float                # plane-wave integerization multiplier
    lambda_: float          # wavelength
    K: int                  # number of plane waves
    axis_avoid_deg: float = 20.0


@dataclass
class Geometry:
    positions: np.ndarray       # (N,3) float64 [l,m,n]
    velocities: np.ndarray      # (N,3) complex128 [vx,vy,vz]
    meta: Dict[str, object]


@dataclass
class PlaneWaveSet:
    pqr: np.ndarray             # (K,3) int32
    octant_counts: Tuple[int, ...]


# =============================
# Pulsating sphere geometry
# =============================

def spherical_fibonacci_points(N: int, radius: float):
    """
    Generate N points on a sphere of given radius using the spherical
    Fibonacci method, plus associated unit radial directions.

    Returns:
      pts  : (N,3) positions [l,m,n]
      dirs : (N,3) unit radial vectors (radial directions)
    """
    i = np.arange(N, dtype=np.float64)
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    ga = 2.0 * np.pi * (1.0 - 1.0 / phi)

    z = 1.0 - 2.0 * (i + 0.5) / N
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    theta = ga * i

    x = r * np.cos(theta)
    y = r * np.sin(theta)

    pts_unit = np.stack([x, y, z], axis=1)
    pts_unit /= np.linalg.norm(pts_unit, axis=1, keepdims=True)

    pts = radius * pts_unit
    return pts, pts_unit


def build_geometry_from_sphere(N: int, radius: float) -> Geometry:
    """
    Convenience: build Geometry for a pulsating sphere with
    unit radial velocity at each point.
    """
    pts, dirs = spherical_fibonacci_points(N, radius)
    vx, vy, vz = dirs[:, 0], dirs[:, 1], dirs[:, 2]

    velocities = np.column_stack([vx, vy, vz]).astype(np.complex128)
    positions = pts.astype(np.float64)

    meta: Dict[str, object] = {
        "units_length": "arbitrary",
        "units_velocity": "arbitrary",
        "radius": radius,
    }
    return Geometry(positions=positions, velocities=velocities, meta=meta)


# =============================
# Plane-wave utilities
# =============================

def spherical_fibonacci(n: int) -> np.ndarray:
    i = np.arange(n, dtype=np.float64)
    phi = (1 + 5**0.5) / 2.0
    ga = 2.0 * np.pi * (1.0 - 1.0 / phi)
    z = 1.0 - 2.0 * (i + 0.5) / n
    r = np.sqrt(np.maximum(0.0, 1.0 - z*z))
    theta = ga * i
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    u = np.stack([x, y, z], axis=1)
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    return u


def within_axis_cone(u: np.ndarray, deg: float) -> np.ndarray:
    if deg <= 0:
        return np.ones(len(u), dtype=bool)
    cos_thr = np.cos(np.deg2rad(deg))
    ux, uy, uz = np.abs(u[:, 0]), np.abs(u[:, 1]), np.abs(u[:, 2])
    bad = (ux >= cos_thr) | (uy >= cos_thr) | (uz >= cos_thr)
    return ~bad


def octant_index(v: np.ndarray) -> np.ndarray:
    s = (v >= 0).astype(np.int32)
    return (s[:, 0] << 2) + (s[:, 1] << 1) + s[:, 2]


def dedupe_triplets(pqr: np.ndarray) -> np.ndarray:
    if len(pqr) == 0:
        return pqr
    a = np.ascontiguousarray(pqr, dtype=np.int32)
    b = a.view([("p", np.int32), ("q", np.int32), ("r", np.int32)])
    _, idx = np.unique(b, return_index=True)
    return a[np.sort(idx)]


def forbid_mirrors(pqr: np.ndarray) -> np.ndarray:
    if len(pqr) == 0:
        return pqr
    a = np.ascontiguousarray(pqr, dtype=np.int32)
    seen = set()
    keep_idx: List[int] = []
    for i, (p, q, r) in enumerate(a):
        if (p, q, r) >= (-p, -q, -r):
            key = (p, q, r)
        else:
            key = (-p, -q, -r)
        if key in seen:
            continue
        seen.add(key)
        keep_idx.append(i)
    return a[np.array(keep_idx, dtype=np.int64)]


def generate_pqr(config: Phase2Config,
                 n_unit_dirs: int = 10000) -> PlaneWaveSet:
    """
    Generate plane-wave integer directions (p,q,r) for given W, K.

    Uses spherical Fibonacci unit vectors, axis avoidance, duplicate and
    mirror removal, then simple octant balancing.
    """
    u = spherical_fibonacci(n_unit_dirs)
    u = u[within_axis_cone(u, config.axis_avoid_deg)]

    pqr = np.rint(config.W * u).astype(np.int32)
    pqr = pqr[np.any(pqr != 0, axis=1)]
    pqr = dedupe_triplets(pqr)
    pqr = forbid_mirrors(pqr)

    oct_idx = octant_index(pqr)
    counts = np.zeros(8, dtype=int)
    selected: List[np.ndarray] = []
    target = (config.K + 7) // 8
    for i, v in enumerate(pqr):
        o = oct_idx[i]
        if counts[o] < target:
            selected.append(v)
            counts[o] += 1
            if len(selected) == config.K:
                break
    if len(selected) < config.K:
        for v in pqr:
            if len(selected) == config.K:
                break
            selected.append(v)
    pqr_sel = np.array(selected[:config.K], dtype=np.int32)

    oct_counts = tuple(np.bincount(octant_index(pqr_sel), minlength=8).tolist())
    return PlaneWaveSet(pqr=pqr_sel, octant_counts=oct_counts)


# =============================
# Phase II operator (g, Λ, Vx,Vy,Vz)
# =============================

def build_g(positions: np.ndarray,
            pqr: np.ndarray,
            W: float,
            lambda_: float) -> np.ndarray:
    """
    g[i,k] = exp( j * 2π/(W λ) * (l_i p_k + m_i q_k + n_i r_k) )
    """
    lmn = positions.astype(np.float64)
    pqr_f = pqr.astype(np.float64)
    scale = 2.0 * np.pi / (W * lambda_)
    dots = lmn @ pqr_f.T
    phases = scale * dots
    g = np.exp(1j * phases)
    return g.astype(np.complex128)


def build_V_component_from_pqr(pqr: np.ndarray, axis: str) -> np.ndarray:
    """
    Build Vx, Vy, or Vz diagonal entries from integer (p,q,r).

    Your rule:
      - Each plane wave is defined by directional cosines (ux,uy,uz)
        on the unit sphere.
      - After integerization, p ≈ W*ux, q ≈ W*uy, r ≈ W*uz.
      - For the x-direction operator:
            Vx_k ≈ 1 / (W*ux_k) ≈ 1 / p_k
        similarly:
            Vy_k ≈ 1 / q_k
            Vz_k ≈ 1 / r_k

    Inputs
    ------
    pqr : (K,3) array of ints/floats
        Columns are [p, q, r] for each plane wave.
    axis : {'x','y','z'}
        Which component to build V for.

    Returns
    -------
    V : (K,) float64
        Diagonal entries for the chosen component.
    """
    pqr = np.asarray(pqr, dtype=np.float64)
    p = pqr[:, 0]
    q = pqr[:, 1]
    r = pqr[:, 2]

    eps = 1e-12  # guard against accidental zeros; axis-avoid should already help

    ax = axis.lower()
    if ax == "x":
        denom = np.where(np.abs(p) > eps, p, np.sign(p) * eps)
    elif ax == "y":
        denom = np.where(np.abs(q) > eps, q, np.sign(q) * eps)
    elif ax == "z":
        denom = np.where(np.abs(r) > eps, r, np.sign(r) * eps)
    else:
        raise ValueError("axis must be 'x', 'y', or 'z'")

    V = 1.0 / denom
    return V.astype(np.float64)


def compute_pressure_phase2(config: Phase2Config,
                            geom: Geometry,
                            pqr_set: PlaneWaveSet) -> Dict[str, np.ndarray]:
    """
    Compute surface pressure using DAM 1.1 Phase II operator, with
    separate Vx, Vy, Vz diagonals as per your description:

      For each component ξ ∈ {x, y, z}:

        A_ξ = (1/N) * diag(Vξ) * diag(Lambda^{-1}) * (g^H v_ξ)

      where:
        - Vx_k ≈ 1 / p_k
        - Vy_k ≈ 1 / q_k
        - Vz_k ≈ 1 / r_k

      Total pressure:
        p = g(A_x + A_y + A_z)

    Returns fields:
      p, p_real, p_imag, p_mag, p_phase, Lambda, Vx, Vy, Vz
    """
    positions = geom.positions
    velocities = geom.velocities
    N = positions.shape[0]

    # Keep config.N in sync, but don't enforce it as a constraint
    if N != config.N:
        config.N = N

    # Build g-matrix, shape (N,K)
    g = build_g(positions, pqr_set.pqr, config.W, config.lambda_)

    # Column norms Λ = diag(g^H g) / N
    Lambda = (g.conj().T @ g).diagonal().real / N
    Lambda_inv = 1.0 / np.maximum(Lambda, 1e-12)

    # Build component-specific V diagonals from integer (p,q,r)
    Vx = build_V_component_from_pqr(pqr_set.pqr, axis="x")
    Vy = build_V_component_from_pqr(pqr_set.pqr, axis="y")
    Vz = build_V_component_from_pqr(pqr_set.pqr, axis="z")

    # Accumulate contribution from x, y, z velocity components
    p_total = np.zeros(N, dtype=np.complex128)

    # Px: use only Vx for vx
    vx = velocities[:, 0]            # (N,)
    proj_x = g.conj().T @ vx         # (K,)
    A_x = (1.0 / N) * Vx * Lambda_inv * proj_x
    p_total += g @ A_x               # (N,)

    # Py: use only Vy for vy
    vy = velocities[:, 1]
    proj_y = g.conj().T @ vy
    A_y = (1.0 / N) * Vy * Lambda_inv * proj_y
    p_total += g @ A_y

    # Pz: use only Vz for vz
    vz = velocities[:, 2]
    proj_z = g.conj().T @ vz
    A_z = (1.0 / N) * Vz * Lambda_inv * proj_z
    p_total += g @ A_z

    p = p_total

    return {
        "p": p,
        "p_real": p.real,
        "p_imag": p.imag,
        "p_mag": np.abs(p),
        "p_phase": np.angle(p),
        "Lambda": Lambda,
        "Vx": Vx,
        "Vy": Vy,
        "Vz": Vz,
    }


# =============================
# Diagnostics (octant-wise)
# =============================

def octant_index_from_positions(positions: np.ndarray) -> np.ndarray:
    """
    Map surface points to octant indices 0..7 based on signs of (l,m,n).
    """
    s = (positions >= 0.0).astype(int)
    return (s[:, 0] << 2) + (s[:, 1] << 1) + s[:, 2]


def compute_octant_stats(positions: np.ndarray,
                         p: np.ndarray) -> pd.DataFrame:
    """
    Compute per-octant mean/STD of Re(p), Im(p), |p|.

    Returns a DataFrame with columns:
      Octant, Count, Re_mean, Im_mean, Mag_mean, Re_std, Im_std, Mag_std
    """
    oct_idx = octant_index_from_positions(positions)
    rows = []
    for o in range(8):
        mask = (oct_idx == o)
        if not mask.any():
            continue
        p_o = p[mask]
        Re = p_o.real
        Im = p_o.imag
        Mag = np.abs(p_o)
        rows.append({
            "Octant": o,
            "Count": int(mask.sum()),
            "Re_mean": float(Re.mean()),
            "Im_mean": float(Im.mean()),
            "Mag_mean": float(Mag.mean()),
            "Re_std": float(Re.std(ddof=1)) if mask.sum() > 1 else 0.0,
            "Im_std": float(Im.std(ddof=1)) if mask.sum() > 1 else 0.0,
            "Mag_std": float(Mag.std(ddof=1)) if mask.sum() > 1 else 0.0,
        })
    return pd.DataFrame(rows)


# =============================
# Streamlit UI
# =============================

st.set_page_config(
    page_title="DAM 1.1 Phase II – Pulsating Sphere",
    layout="wide",
)

st.title("DAM 1.1 Phase II – Pulsating Sphere Surface Pressure")

st.markdown(
    """
This app generates a **pulsating sphere** test case and computes surface
pressure using the DAM 1.1 Phase II operator.

Steps:
- Generate N points on a sphere of radius a using spherical Fibonacci.
- Assign unit radial velocity at each point.
- Generate K plane waves (with W, axis-avoidance, octant balancing).
- Compute pressure via DAM 1.1 Phase II.
"""
)


# Sidebar controls
st.sidebar.header("Sphere & Phase II Parameters")

N = st.sidebar.number_input(
    "N (number of surface points)",
    min_value=4,
    value=80,
    step=1,
    help="Number of points on the spherical radiator surface.",
)

radius = st.sidebar.number_input(
    "Sphere radius a",
    min_value=1.0,
    value=100.0,
    step=1.0,
    help="Radius of the pulsating sphere.",
)

W = st.sidebar.number_input(
    "W (plane-wave multiplier)",
    min_value=1.0,
    value=500.0,
    step=1.0,
    help="Scaling factor before rounding (p,q,r) = round(W * unit_direction).",
)

mode_lambda = st.sidebar.selectbox(
    "How to choose wavelength λ",
    [
        "Specify λ directly",
        "Compute λ from ka=1 (pulsating sphere theory)",
    ],
)
