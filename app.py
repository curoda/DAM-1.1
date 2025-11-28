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
# Phase II operator (g, Λ, V)
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


def build_V_diagonal(pqr: np.ndarray) -> np.ndarray:
    """
    Build V_k from inverse cosines of direction cosines of each plane-wave direction.

    Example scalar:
      V_k = (arccos(ux) + arccos(uy) + arccos(uz)) / 3
    """
    pqr_f = pqr.astype(np.float64)
    norms = np.linalg.norm(pqr_f, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    u = pqr_f / norms

    ux = np.clip(u[:, 0], -1.0, 1.0)
    uy = np.clip(u[:, 1], -1.0, 1.0)
    uz = np.clip(u[:, 2], -1.0, 1.0)

    theta_x = np.arccos(ux)
    theta_y = np.arccos(uy)
    theta_z = np.arccos(uz)

    V = (theta_x + theta_y + theta_z) / 3.0
    return V.astype(np.float64)


def compute_pressure_phase2(config: Phase2Config,
                            geom: Geometry,
                            pqr_set: PlaneWaveSet) -> Dict[str, np.ndarray]:
    """
    Compute surface pressure using DAM 1.1 Phase II operator:

      A_x = (1/N) * diag(V) * diag(Lambda^{-1}) * (g^H v_x)
      A_y = ...
      A_z = ...
      p = g(A_x + A_y + A_z)

    Returns fields:
      p, p_real, p_imag, p_mag, p_phase, Lambda, V
    """
    positions = geom.positions
    velocities = geom.velocities
    N = positions.shape[0]

    if N != config.N:
        config.N = N

    g = build_g(positions, pqr_set.pqr, config.W, config.lambda_)
    Lambda = (g.conj().T @ g).diagonal().real / N
    Lambda_inv = 1.0 / np.maximum(Lambda, 1e-12)

    V = build_V_diagonal(pqr_set.pqr)

    p_total = np.zeros(N, dtype=np.complex128)
    for comp in range(3):
        vcomp = velocities[:, comp]
        proj = g.conj().T @ vcomp
        A = (1.0 / N) * V * Lambda_inv * proj
        p_total += g @ A

    p = p_total
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

if mode_lambda == "Specify λ directly":
    lambda_val = st.sidebar.number_input(
        "λ (wavelength)",
        min_value=1e-9,
        value=2.0 * math.pi * radius,
        step=1.0,
        format="%.6f",
    )
else:
    # ka = 1 → k = 1/a → λ = 2πa
    k = 1.0 / radius
    lambda_val = 2.0 * math.pi / k
    st.sidebar.markdown(
        f"Computed from ka = 1 and radius a = {radius:.3f}: "
        f"λ ≈ {lambda_val:.3f}"
    )

default_K = 3 * N
predefined_Ks = sorted(set([800, 1600, 2400, 3200, default_K]))
K = st.sidebar.selectbox(
    "K (number of plane waves)",
    options=predefined_Ks,
    index=predefined_Ks.index(default_K) if default_K in predefined_Ks else len(predefined_Ks) - 1,
    help="3N is a common default; other choices are 800,1600,2400,3200.",
)

axis_avoid = st.sidebar.number_input(
    "Axis avoid angle (deg)",
    min_value=0.0,
    max_value=89.9,
    value=20.0,
    step=1.0,
    help="Cone half-angle about coordinate axes to avoid plane waves close to axes.",
)

run_button = st.sidebar.button("Run Phase II computation")


# Generate geometry preview
pts, dirs = spherical_fibonacci_points(N, radius)
l, m, n = pts[:, 0], pts[:, 1], pts[:, 2]

df_geom_preview = pd.DataFrame({
    "l": l,
    "m": m,
    "n": n,
    "vx": dirs[:, 0],
    "vy": dirs[:, 1],
    "vz": dirs[:, 2],
})

st.subheader("Generated geometry and radial velocity (preview)")
st.write(f"Sphere radius = {radius}, N = {N}")
st.dataframe(df_geom_preview.head())

with st.expander("Show 3D geometry scatter (positions only)"):
    fig_geo = plt.figure(figsize=(5, 5))
    axg = fig_geo.add_subplot(111, projection="3d")
    axg.scatter(l, m, n)
    axg.set_xlabel("l")
    axg.set_ylabel("m")
    axg.set_zlabel("n")
    axg.set_title("Generated sphere points")
    max_range = np.array([l.max()-l.min(), m.max()-m.min(), n.max()-n.min()]).max() / 2.0
    mid_x = 0.5 * (l.max() + l.min())
    mid_y = 0.5 * (m.max() + m.min())
    mid_z = 0.5 * (n.max() + n.min())
    axg.set_xlim(mid_x - max_range, mid_x + max_range)
    axg.set_ylim(mid_y - max_range, mid_y + max_range)
    axg.set_zlim(mid_z - max_range, mid_z + max_range)
    st.pyplot(fig_geo)


if not run_button:
    st.stop()

st.info("Running DAM 1.1 Phase II with generated pulsating-sphere data...")

geom = build_geometry_from_sphere(N, radius)
config = Phase2Config(
    N=N,
    W=W,
    lambda_=lambda_val,
    K=K,
    axis_avoid_deg=axis_avoid,
)

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


# Visualization of |p|
st.subheader("|p| distribution (3D scatter)")

fig = plt.figure(figsize=(5, 5))
ax = fig.add_subplot(111, projection="3d")
pos = geom.positions
p_mag = result["p_mag"]

size_min = 10.0
size_max = 80.0
if p_mag.max() > 0:
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
