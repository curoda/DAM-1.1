#!/usr/bin/env python3

from __future__ import annotations

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

def spherical_fibonacci_points(M: int, radius: float = 1.0) -> np.ndarray:
    """
    Generate M approximately uniform points on a sphere of given radius
    using the spherical Fibonacci / golden-angle method.
    Returns array of shape (M,3) with x,y,z coordinates.
    """
    # Standard spherical Fibonacci construction
    k = np.arange(M, dtype=np.float64)
    phi = (1 + 5 ** 0.5) / 2  # golden ratio
    ga = 2.0 * np.pi / phi

    z = 1.0 - (2.0 * k + 1.0) / M
    r = np.sqrt(np.maximum(1.0 - z * z, 0.0))
    theta = ga * k

    x = r * np.cos(theta)
    y = r * np.sin(theta)

    pts = np.stack([x, y, z], axis=1)
    pts *= float(radius)
    return pts


def build_geometry_from_sphere(N: int, radius: float = 100.0) -> Geometry:
    """
    Build Geometry for pulsating sphere:
      - N nearly-uniform points on radius=radius sphere
      - integerised positions (rounded)
      - velocities = unit radial directions (complex128)
    """
    dirs = spherical_fibonacci_points(N, radius=1.0)
    # Integerised positions at radius
    pos_cont = dirs * radius
    lmn_int = np.rint(pos_cont).astype(np.float64)

    # Unit radial velocities (same as radius=1 points)
    vx, vy, vz = dirs[:, 0], dirs[:, 1], dirs[:, 2]
    velocities = np.column_stack([vx, vy, vz]).astype(np.complex128)

    meta = {
        "radius": radius,
        "dirs_unit": dirs,
    }

    return Geometry(positions=lmn_int, velocities=velocities, meta=meta)


# =============================
# Plane-wave / pqr generation
# =============================

def spherical_coordinates_from_cartesian(x: np.ndarray,
                                         y: np.ndarray,
                                         z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert Cartesian (x,y,z) to spherical (theta, phi):
      theta: polar angle [0,pi]
      phi: azimuthal angle [-pi,pi]
    """
    r = np.sqrt(x * x + y * y + z * z)
    r = np.maximum(r, 1e-12)
    theta = np.arccos(np.clip(z / r, -1.0, 1.0))
    phi = np.arctan2(y, x)
    return theta, phi


def cartesian_from_spherical(theta: np.ndarray,
                             phi: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert spherical (theta, phi) to Cartesian direction cosines (x,y,z) with r=1.
    """
    sin_t = np.sin(theta)
    x = sin_t * np.cos(phi)
    y = sin_t * np.sin(phi)
    z = np.cos(theta)
    return x, y, z


def generate_plane_wave_directions(K: int,
                                   axis_avoid_deg: float = 20.0) -> np.ndarray:
    """
    Generate K unit-vector plane-wave directions approximately uniformly over the sphere,
    with axis-avoidance around +/- x,y,z of given half-angle in degrees.

    Returns array of shape (K,3) of floats [ux,uy,uz].
    """
    # Use an oversampled Fibonacci set, then filter and pick K directions
    oversample_factor = 6
    M = max(K * oversample_factor, K + 64)
    dirs = spherical_fibonacci_points(M, radius=1.0)
    x, y, z = dirs[:, 0], dirs[:, 1], dirs[:, 2]

    # Axis-avoidance: reject directions within axis_avoid_deg of +/- x,y,z
    axis_avoid_rad = np.deg2rad(axis_avoid_deg)
    cos_avoid = np.cos(axis_avoid_rad)
    keep_mask = (
        (np.abs(x) < cos_avoid) &
        (np.abs(y) < cos_avoid) &
        (np.abs(z) < cos_avoid)
    )
    dirs_kept = dirs[keep_mask]

    if dirs_kept.shape[0] < K:
        # If we rejected too many, relax criterion slightly
        axis_avoid_rad = np.deg2rad(axis_avoid_deg * 0.8)
        cos_avoid = np.cos(axis_avoid_rad)
        keep_mask = (
            (np.abs(x) < cos_avoid) &
            (np.abs(y) < cos_avoid) &
            (np.abs(z) < cos_avoid)
        )
        dirs_kept = dirs[keep_mask]

    if dirs_kept.shape[0] < K:
        raise RuntimeError(
            f"Unable to generate {K} axis-avoiding directions; only {dirs_kept.shape[0]} candidates."
        )

    # For simplicity: just take first K after filtering
    dirs_sel = dirs_kept[:K]
    return dirs_sel


def octant_index(vecs: np.ndarray) -> np.ndarray:
    """
    Compute octant index 0..7 for each row of vecs shaped (K,3).
    Octant coding:
      bit0: x>=0
      bit1: y>=0
      bit2: z>=0
    """
    x_pos = vecs[:, 0] >= 0
    y_pos = vecs[:, 1] >= 0
    z_pos = vecs[:, 2] >= 0
    return (x_pos.astype(int)
            + 2 * y_pos.astype(int)
            + 4 * z_pos.astype(int))


def balance_octants(dirs: np.ndarray, K: int) -> np.ndarray:
    """
    Given candidate directions dirs (M,3), pick K with balanced octant counts
    as much as possible.
    """
    if dirs.shape[0] <= K:
        return dirs

    octs = octant_index(dirs)
    # desired per-octant count (roughly)
    base = K // 8
    remainder = K % 8
    desired = np.array([base + (1 if i < remainder else 0) for i in range(8)], dtype=int)

    selected_indices: List[int] = []
    used_per_oct = np.zeros(8, dtype=int)

    for i in range(dirs.shape[0]):
        o = octs[i]
        if used_per_oct[o] < desired[o]:
            selected_indices.append(i)
            used_per_oct[o] += 1
        if len(selected_indices) == K:
            break

    # If we didn't hit exactly K, just pad from remaining
    if len(selected_indices) < K:
        remaining = [i for i in range(dirs.shape[0]) if i not in selected_indices]
        needed = K - len(selected_indices)
        selected_indices.extend(remaining[:needed])

    selected_indices = np.array(selected_indices, dtype=int)
    return dirs[selected_indices]


def integerize_directions(dirs: np.ndarray, W: float) -> np.ndarray:
    """
    Multiply direction cosines by W and round to nearest integer.
    Returns (K,3) int32 array [p,q,r].
    """
    scaled = dirs * float(W)
    pqr = np.rint(scaled).astype(np.int32)
    return pqr


def generate_pqr(config: Phase2Config) -> PlaneWaveSet:
    """
    Generate K plane-wave directions and integerised p,q,r, with octant balancing.
    """
    dirs = generate_plane_wave_directions(config.K, axis_avoid_deg=config.axis_avoid_deg)
    dirs_bal = balance_octants(dirs, config.K)
    pqr = integerize_directions(dirs_bal, config.W)

    octs = octant_index(dirs_bal)
    counts = tuple(int(np.sum(octs == i)) for i in range(8))

    return PlaneWaveSet(pqr=pqr, octant_counts=counts)


# =============================
# Core operator: g, V, Lambda
# =============================

def build_g(positions: np.ndarray,
            pqr: np.ndarray,
            W: float,
            lambda_: float) -> np.ndarray:
    lmn = positions.astype(np.float64)
    pqr_f = pqr.astype(np.float64)
    scale = 2.0 * np.pi / (W * lambda_)
    dots = lmn @ pqr_f.T
    phases = scale * dots
    g = np.exp(1j * phases)
    return g.astype(np.complex128)


def build_V_components(pqr: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build per-component diagonal weights Vx, Vy, Vz from integerized plane-wave
    directions (p, q, r).

    For each plane wave k we use

        Vx_k ≈ 1 / p_k
        Vy_k ≈ 1 / q_k
        Vz_k ≈ 1 / r_k

    where (p_k, q_k, r_k) are the integer directions obtained by multiplying
    the direction cosines by W and rounding.

    Any zero entries are guarded with a small epsilon to avoid division by zero.
    """
    pqr_f = pqr.astype(np.float64)
    p = pqr_f[:, 0]
    q = pqr_f[:, 1]
    r = pqr_f[:, 2]

    eps = 1e-12
    Vx = np.zeros_like(p, dtype=np.float64)
    Vy = np.zeros_like(q, dtype=np.float64)
    Vz = np.zeros_like(r, dtype=np.float64)

    mask_p = np.abs(p) > eps
    mask_q = np.abs(q) > eps
    mask_r = np.abs(r) > eps

    Vx[mask_p] = 1.0 / p[mask_p]
    Vy[mask_q] = 1.0 / q[mask_q]
    Vz[mask_r] = 1.0 / r[mask_r]

    # Any entries with |p|,|q|,|r| ≈ 0 are left at 0. The axis-avoidance rules
    # in the plane-wave generator should prevent such cases in normal use.
    return Vx, Vy, Vz


def build_V_diagonal(pqr: np.ndarray) -> np.ndarray:
    """
    Backwards-compatible helper that returns a (K,3) array stacking
    Vx, Vy, Vz for each plane wave.
    """
    Vx, Vy, Vz = build_V_components(pqr)
    return np.column_stack([Vx, Vy, Vz])


def compute_pressure_phase2(config: Phase2Config,
                            geom: Geometry,
                            pqr_set: PlaneWaveSet) -> Dict[str, np.ndarray]:
    """
    Compute surface pressure using DAM 1.1 Phase II operator with
    per-component diagonal weights:

      A_x = (1/N) * diag(Vx) * diag(Lambda^{-1}) * (g^H v_x)
      A_y = (1/N) * diag(Vy) * diag(Lambda^{-1}) * (g^H v_y)
      A_z = (1/N) * diag(Vz) * diag(Lambda^{-1}) * (g^H v_z)
      p   = g (A_x + A_y + A_z)

    Returns fields:
      p, p_real, p_imag, p_mag, p_phase, Lambda, V

    where V is a (K,3) array stacking [Vx, Vy, Vz] for each plane wave.
    """
    positions = geom.positions
    velocities = geom.velocities

    N = positions.shape[0]

    # Build propagation matrix g and its Gram diagonal Lambda
    g = build_g(positions, pqr_set.pqr, config.W, config.lambda_)
    Lambda = (g.conj().T @ g).diagonal().real / N
    Lambda_inv = 1.0 / np.maximum(Lambda, 1e-12)

    # Build per-component diagonal weights from integerised (p, q, r)
    Vx, Vy, Vz = build_V_components(pqr_set.pqr)

    p_total = np.zeros(N, dtype=np.complex128)

    for comp, Vcomp in zip(range(3), (Vx, Vy, Vz)):
        vcomp = velocities[:, comp]
        proj = g.conj().T @ vcomp
        A = (1.0 / N) * Vcomp * Lambda_inv * proj
        p_total += g @ A

    p = p_total
    V = np.column_stack([Vx, Vy, Vz])

    return {
        "p": p,
        "p_real": p.real,
        "p_imag": p.imag,
        "p_mag": np.abs(p),
        "p_phase": np.angle(p),
        "Lambda": Lambda,
        "V": V,
    }


# =============================
# Streamlit app
# =============================

st.set_page_config(
    page_title="DAM 1.1 Phase II – Pulsating Sphere",
    layout="centered",
)

st.title("DAM 1.1 Phase II – Pulsating Sphere Test")

st.markdown(
    """
Interactive prototype to exercise the DAM 1.1 Phase II operator on a pulsating
sphere geometry.

- Generate N points on a radius=a sphere (integerised l,m,n).
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
    step=4,
)

radius = st.sidebar.number_input(
    "Sphere radius a",
    min_value=1.0,
    value=100.0,
    step=10.0,
)

W = st.sidebar.number_input(
    "W (direction cosine integerization multiplier)",
    min_value=10.0,
    value=1000.0,
    step=10.0,
)

lambda_mode = st.sidebar.selectbox(
    "Wavelength λ",
    options=["Specify λ directly", "Compute λ from ka=1 (pulsating sphere theory)"],
    index=1,
)

if lambda_mode == "Specify λ directly":
    lambda_val = st.sidebar.number_input(
        "λ (wavelength)",
        min_value=1.0,
        value=680.0,
        step=10.0,
    )
else:
    # ka = 1 => k = 1/a, so λ = 2π / k = 2π a
    k = 1.0 / radius
    lambda_val = 2.0 * math.pi / k
    st.sidebar.info(f"ka=1 ⇒ λ = 2πa ≈ {lambda_val:.4g}")

K = st.sidebar.number_input(
    "K (number of plane waves)",
    min_value=8,
    value=80,
    step=8,
)

axis_avoid = st.sidebar.slider(
    "Axis-avoid angle (deg)",
    min_value=0.0,
    max_value=45.0,
    value=20.0,
    step=1.0,
)

config = Phase2Config(
    N=int(N),
    W=float(W),
    lambda_=float(lambda_val),
    K=int(K),
    axis_avoid_deg=float(axis_avoid),
)

st.write("## Step 1: Geometry and velocities")

with st.spinner("Generating pulsating sphere geometry..."):
    geom = build_geometry_from_sphere(config.N, radius=radius)

st.success("Geometry generated.")

st.write(f"Generated {geom.positions.shape[0]} integerised points on a radius={radius} sphere.")

if st.checkbox("Show first 10 geometry points"):
    df_geom = pd.DataFrame(
        np.column_stack([geom.positions, geom.velocities.real, geom.velocities.imag]),
        columns=["l", "m", "n", "vx_real", "vy_real", "vz_real", "vx_imag", "vy_imag", "vz_imag"],
    )
    st.dataframe(df_geom.head(10))


st.write("## Step 2: Plane waves and Phase II pressure")

with st.spinner("Generating plane waves and computing pressure..."):
    pqr_set = generate_pqr(config)
    result = compute_pressure_phase2(config, geom, pqr_set)

st.success("Computation complete.")


# Summary
st.subheader("Configuration summary")

col1, col2, col3 = st.columns(3)
with col1:
    st.write(f"N = {geom.positions.shape[0]}")
    st.write(f"Radius a = {radius:.6g}")
with col2:
    st.write(f"K = {config.K}")
    st.write(f"W = {config.W}")
with col3:
    st.write(f"λ = {config.lambda_:.6g}")
    st.write(f"Axis avoid = {config.axis_avoid_deg}°")

st.write(f"Plane-wave octant counts = {pqr_set.octant_counts}")

Lambda = result["Lambda"]
st.write(
    f"Λ diag stats: min = {Lambda.min():.6g}, "
    f"max = {Lambda.max():.6g}, "
    f"mean = {Lambda.mean():.6g}"
)


# Pressure table and download
st.subheader("Pressure results (first 10 points)")

out_df = pd.DataFrame({
    "l": geom.positions[:, 0],
    "m": geom.positions[:, 1],
    "n": geom.positions[:, 2],
    "p_real": result["p_real"],
    "p_imag": result["p_imag"],
    "p_mag": result["p_mag"],
    "p_phase": result["p_phase"],
})

st.dataframe(out_df.head(10))

csv_bytes = out_df.to_csv(index=False).encode("utf-8")
st.download_button(
    label="Download full pressure CSV",
    data=csv_bytes,
    file_name="phase2_pulsating_sphere_pressures.csv",
    mime="text/csv",
)


# Optional 3D scatter of |p|
st.subheader("3D view of |p| on the sphere")

fig = plt.figure(figsize=(5, 5))
ax = fig.add_subplot(111, projection="3d")

pos = geom.positions
p_mag = result["p_mag"]

# Scale marker sizes by |p|
size_min, size_max = 10.0, 80.0
if np.ptp(p_mag) > 1e-12:
    sizes = size_min + (size_max - size_min) * (p_mag - p_mag.min()) / max(
        p_mag.max() - p_mag.min(), 1e-12
    )
else:
    sizes = np.full_like(p_mag, size_min)

ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], s=sizes)
ax.set_xlabel("l")
ax.set_ylabel("m")
ax.set_zlabel("n")
ax.set_title("|p| on pulsating sphere")
ax.set_box_aspect([1, 1, 1])

st.pyplot(fig)
