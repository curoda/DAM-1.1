#!/usr/bin/env python3
"""
Streamlit app: DAM 1.2 – Pulsating Sphere Test

This app keeps the same Streamlit interface as the prior prototype, but updates the
core logic to match:
- Sphere_Data_1_1 (geometry + outward-normal velocities for a pulsating sphere)
- DAM_1_2 (plane-wave selection rules + pressure computation)

Key updates vs the older app.py:
- Geometry (l,m,n) is NOT rounded to integers; it is R * unit normal.
- Velocities (vx,vy,vz) are the unit normal (not multiplied by R).
- Plane waves enforce:
    * axis-avoidance (default 20°) from x/y/z axes
    * NO mirror images: never include both (p,q,r) and (-p,-q,-r)
    * balanced octants (K/8 per octant, with remainder distributed)
    * reject any candidate that would round to p==0 or q==0 or r==0
- Pressure computation follows DAM 1.2 (no Lambda inverse term).
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st


# =============================
# Data models
# =============================

@dataclass(frozen=True)
class Phase2Config:
    N: int                  # number of geometry points
    W: float                # plane-wave integerization multiplier
    lambda_: float          # wavelength
    K: int                  # number of plane waves
    axis_avoid_deg: float = 20.0
    candidates: int = 4000
    seed: int = 1


@dataclass
class Geometry:
    positions: np.ndarray       # (N,3) float64 [l,m,n]
    velocities: np.ndarray      # (N,3) complex128 [vx,vy,vz] (imag=0 for this test)
    meta: Dict[str, object]


@dataclass
class PlaneWaveSet:
    pqr: np.ndarray                 # (K,3) int64
    octant_counts: Tuple[int, ...]  # length 8
    mirror_pair_count: int          # should be 0


# =============================
# Fibonacci sphere (unit vectors)
# =============================

def fibonacci_sphere_points(npts: int) -> np.ndarray:
    """
    Unit-sphere points using Fibonacci / golden-angle method.
    Matches the approach used in Sphere_Data_1_1 and DAM_1_2.
    """
    if npts <= 0:
        raise ValueError("npts must be positive")

    ga = math.pi * (3.0 - math.sqrt(5.0))  # golden angle
    pts = np.zeros((npts, 3), dtype=float)

    # The (i+0.5)/npts centering avoids points exactly at the poles.
    for i in range(npts):
        z = 1.0 - 2.0 * (i + 0.5) / npts
        r = math.sqrt(max(0.0, 1.0 - z * z))
        phi = ga * (i + 0.5)
        x = r * math.cos(phi)
        y = r * math.sin(phi)
        pts[i] = (x, y, z)

    pts /= np.linalg.norm(pts, axis=1)[:, None]
    return pts


# =============================
# Pulsating sphere geometry
# =============================

def build_geometry_from_sphere(N: int, radius: float = 100.0) -> Geometry:
    """
    Build Geometry for a pulsating sphere:
      - N nearly-uniform points on a radius=radius sphere:
            (l,m,n) = radius * unit_normal
      - outward normal velocity components are the unit normal:
            (vx,vy,vz) = unit_normal
    """
    unit_dirs = fibonacci_sphere_points(int(N))              # (N,3) on unit sphere
    geom = float(radius) * unit_dirs                         # (N,3) geometry
    vel = unit_dirs                                          # (N,3) velocity components (unit)

    meta = {"radius": float(radius), "dirs_unit": unit_dirs}
    return Geometry(
        positions=geom.astype(np.float64),
        velocities=vel.astype(np.complex128),
        meta=meta,
    )


# =============================
# Plane-wave / pqr selection (DAM 1.2 rules)
# =============================

def octant_index(vecs: np.ndarray) -> np.ndarray:
    """
    Compute octant index 0..7 using sign bits:
      bit0: x>=0
      bit1: y>=0
      bit2: z>=0
    """
    x_pos = vecs[:, 0] >= 0
    y_pos = vecs[:, 1] >= 0
    z_pos = vecs[:, 2] >= 0
    return (x_pos.astype(int) + 2 * y_pos.astype(int) + 4 * z_pos.astype(int))


def _count_mirror_pairs(pqr: np.ndarray) -> int:
    s = {tuple(row.tolist()) for row in pqr}
    pairs = 0
    for t in s:
        if (-t[0], -t[1], -t[2]) in s:
            pairs += 1
    return pairs // 2


def select_plane_waves_pqr(
    K: int,
    W: float,
    axis_avoid_deg: float = 20.0,
    candidate_M: int = 4000,
    seed: int = 1,
    max_candidates: int = 200000,
    max_attempts: int = 8,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Select K plane-wave directions as integer triplets (p,q,r) using DAM 1.2 rules:

      - reject any direction within axis_avoid_deg of ANY axis (x, y, z)
      - enforce NO mirror images: do not include both (p,q,r) and (-p,-q,-r)
      - balance by octant (K/8 per octant, remainder distributed)
      - scale by W and round -> integer p,q,r
      - reject any candidate with p==0 or q==0 or r==0
      - keep unique rounded triplets in the candidate pool

    Returns:
      pqr_int: (K,3) int array
      dirs_sel: (K,3) float unit vectors corresponding to the selected pqr
    """
    K = int(K)
    if K < 8:
        raise ValueError("K must be at least 8 to balance octants")
    if candidate_M < K:
        candidate_M = K

    rnd = random.Random(int(seed))
    cos0 = math.cos(math.radians(float(axis_avoid_deg)))

    base = K // 8
    rem = K % 8
    need = [base + (1 if o < rem else 0) for o in range(8)]

    # Try a few times with an expanding candidate pool to avoid pathological cases.
    for attempt in range(max_attempts):
        M = int(candidate_M * (1.5 ** attempt))

        while True:
            cand = fibonacci_sphere_points(M)
            order = list(range(M))
            rnd.shuffle(order)

            pools: List[List[Tuple[np.ndarray, Tuple[int, int, int]]]] = [[] for _ in range(8)]
            seen_pqr: set[Tuple[int, int, int]] = set()

            for idx in order:
                u = cand[idx]
                x, y, z = float(u[0]), float(u[1]), float(u[2])

                # axis-avoid: reject if within axis_avoid_deg of ANY axis
                if abs(x) > cos0 or abs(y) > cos0 or abs(z) > cos0:
                    continue

                p = int(round(W * x))
                q = int(round(W * y))
                r = int(round(W * z))

                # avoid zero (would make Vx=W/p undefined)
                if p == 0 or q == 0 or r == 0:
                    continue

                pqr = (p, q, r)
                if pqr in seen_pqr:
                    continue
                seen_pqr.add(pqr)

                pools[int(octant_index(u.reshape(1, 3))[0])].append((u, pqr))

            if all(len(pools[o]) >= need[o] for o in range(8)):
                break

            M = int(M * 1.5) + 1
            if M > max_candidates:
                raise RuntimeError(
                    "Could not find enough plane-wave candidates. "
                    "Relax axis-avoid angle, increase W, or increase candidates."
                )

        # Greedy farthest-point selection within each octant, with global mirror constraint.
        global_selected: set[Tuple[int, int, int]] = set()
        selected: List[Tuple[np.ndarray, Tuple[int, int, int]]] = []

        def greedy_farthest(pool, kneed):
            if kneed <= 0:
                return []

            local: List[Tuple[np.ndarray, Tuple[int, int, int]]] = []

            # Pick a valid start
            start_j = None
            for j, (_, pqr) in enumerate(pool):
                if pqr in global_selected:
                    continue
                if (-pqr[0], -pqr[1], -pqr[2]) in global_selected:
                    continue
                start_j = j
                break
            if start_j is None:
                raise RuntimeError("No valid starting direction in an octant pool.")

            u0, pqr0 = pool.pop(start_j)
            local.append((u0, pqr0))
            global_selected.add(pqr0)

            while len(local) < kneed:
                best_j = None
                best_score = -1.0

                for j, (u, pqr) in enumerate(pool):
                    if pqr in global_selected:
                        continue
                    if (-pqr[0], -pqr[1], -pqr[2]) in global_selected:
                        continue

                    # score = minimum separation to already-selected vectors in this octant
                    # separation proxy: 1 - dot(u, us)
                    min_sep = float("inf")
                    for us, _ in local:
                        dot = float(u[0] * us[0] + u[1] * us[1] + u[2] * us[2])
                        sep = 1.0 - dot
                        if sep < min_sep:
                            min_sep = sep

                    if min_sep > best_score:
                        best_score = min_sep
                        best_j = j

                if best_j is None:
                    raise RuntimeError("Ran out of candidates before satisfying octant quota.")

                u_best, pqr_best = pool.pop(best_j)
                local.append((u_best, pqr_best))
                global_selected.add(pqr_best)

            return local

        try:
            for o in range(8):
                pool = pools[o]
                rnd.shuffle(pool)
                selected.extend(greedy_farthest(pool, need[o]))

            if len(selected) != K:
                raise RuntimeError(f"Selected {len(selected)} plane waves, expected {K}.")

            dirs_sel = np.array([u for (u, _) in selected], dtype=float)
            pqr_int = np.array([pqr for (_, pqr) in selected], dtype=int)

            # Final safety check: no mirror pairs
            if _count_mirror_pairs(pqr_int) != 0:
                raise RuntimeError("Mirror-pair constraint violated; retrying selection.")

            return pqr_int, dirs_sel

        except RuntimeError:
            # retry with larger M on the next attempt
            continue

    raise RuntimeError(
        "Failed to select plane waves after multiple attempts. "
        "Try relaxing axis-avoid angle or increasing candidates."
    )


def generate_pqr(config: Phase2Config) -> PlaneWaveSet:
    """
    Generate p,q,r plane-wave triplets using DAM 1.2 selection rules.
    """
    pqr, dirs_sel = select_plane_waves_pqr(
        K=config.K,
        W=config.W,
        axis_avoid_deg=config.axis_avoid_deg,
        candidate_M=config.candidates,
        seed=config.seed,
    )
    octs = octant_index(dirs_sel)
    counts = tuple(int(np.sum(octs == i)) for i in range(8))
    mirror_pairs = _count_mirror_pairs(pqr)
    return PlaneWaveSet(pqr=pqr, octant_counts=counts, mirror_pair_count=mirror_pairs)


# =============================
# Core operator: g and DAM 1.2 pressure
# =============================

def build_g(positions: np.ndarray,
            pqr: np.ndarray,
            W: float,
            lambda_: float) -> np.ndarray:
    """
    g[i,k] = exp( j * 2*pi/(W*lambda) * (l_i p_k + m_i q_k + n_i r_k) )
    """
    lmn = positions.astype(np.float64)
    pqr_f = pqr.astype(np.float64)
    scale = 2.0 * np.pi / (float(W) * float(lambda_))
    phases = scale * (lmn @ pqr_f.T)
    g = np.exp(1j * phases)
    return g.astype(np.complex128)


def compute_pressure_dam_1_2(
    config: Phase2Config,
    geom: Geometry,
    pqr_set: PlaneWaveSet,
) -> Dict[str, np.ndarray]:
    """
    DAM 1.2 pressure computation:

      g[i,k] = exp( j * 2*pi/(W*lambda) * (l_i p_k + m_i q_k + n_i r_k) )
      gH = g^H

      Ax = (1/N) * (W/p) * (gH @ vx)
      Ay = (1/N) * (W/q) * (gH @ vy)
      Az = (1/N) * (W/r) * (gH @ vz)

      p = g@Ax + g@Ay + g@Az

    Notes:
    - No Lambda^{-1} term is used.
    - No orthogonalization of g is attempted.
    """
    positions = geom.positions
    velocities = geom.velocities

    N = int(positions.shape[0])
    pqr = pqr_set.pqr

    g = build_g(positions, pqr, config.W, config.lambda_)
    gH = g.conj().T

    # velocities (N,)
    vx = velocities[:, 0]
    vy = velocities[:, 1]
    vz = velocities[:, 2]

    # p,q,r (K,)
    p = pqr[:, 0].astype(np.float64)
    q = pqr[:, 1].astype(np.float64)
    r = pqr[:, 2].astype(np.float64)

    eps = 1e-12
    Wx = np.where(np.abs(p) > eps, float(config.W) / p, 0.0)
    Wy = np.where(np.abs(q) > eps, float(config.W) / q, 0.0)
    Wz = np.where(np.abs(r) > eps, float(config.W) / r, 0.0)

    Ax = Wx * (gH @ vx) / N
    Ay = Wy * (gH @ vy) / N
    Az = Wz * (gH @ vz) / N

    p_total = (g @ Ax) + (g @ Ay) + (g @ Az)

    p_real = p_total.real.astype(np.float64)
    p_imag = p_total.imag.astype(np.float64)
    p_mag = np.abs(p_total).astype(np.float64)
    p_phase = np.angle(p_total).astype(np.float64)

    # Diagnostics similar to DAM_1_2 script output
    mag = p_mag
    delta_mag = float((mag.max() - mag.min()) / (mag.mean() + 1e-30))
    delta_phase = float(p_phase.max() - p_phase.min())

    return {
        "p": p_total,
        "p_real": p_real,
        "p_imag": p_imag,
        "p_mag": p_mag,
        "p_phase": p_phase,
        "delta_mag": np.array(delta_mag),
        "delta_phase": np.array(delta_phase),
        "pqr": pqr,
        "Wx": Wx,
        "Wy": Wy,
        "Wz": Wz,
    }


# =============================
# Streamlit app
# =============================

st.set_page_config(
    page_title="DAM 1.2 – Pulsating Sphere",
    layout="centered",
)

st.title("DAM 1.2 – Pulsating Sphere Test")

st.markdown(
    """
Interactive prototype to exercise DAM 1.2 logic on a pulsating sphere geometry.

- Generate **N** points on a radius **a** sphere: *(l,m,n) = a · unit_normal* (no rounding).
- Assign outward normal velocity at each point: *(vx,vy,vz) = unit_normal*.
- Generate **K** plane waves with:
  - axis-avoidance (default 20°),
  - **no mirror pairs** (never both pqr and -pqr),
  - balanced octants,
  - rejection of any p,q,r that round to 0.
- Compute pressure via **DAM 1.2** (no Λ^{-1} term).
"""
)

# Sidebar controls
st.sidebar.header("Sphere & DAM 1.2 Parameters")

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
    k = 1.0 / float(radius)
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

with st.sidebar.expander("Advanced plane-wave selection"):
    candidates = st.number_input(
        "Candidate directions (M)",
        min_value=K,
        value=max(4000, int(K) * 10),
        step=500,
        help="Initial candidate pool size. The selector may increase this automatically if needed.",
    )
    seed = st.number_input(
        "Random seed",
        min_value=0,
        value=1,
        step=1,
    )

config = Phase2Config(
    N=int(N),
    W=float(W),
    lambda_=float(lambda_val),
    K=int(K),
    axis_avoid_deg=float(axis_avoid),
    candidates=int(candidates),
    seed=int(seed),
)

st.write("## Step 1: Geometry and velocities")

with st.spinner("Generating pulsating sphere geometry..."):
    geom = build_geometry_from_sphere(config.N, radius=float(radius))

st.success("Geometry generated.")

st.write(f"Generated {geom.positions.shape[0]} points on a radius={float(radius):.6g} sphere.")

# Geometry CSV download (Sphere_Data_1_1 format)
geom_df = pd.DataFrame({
    "idx": np.arange(1, geom.positions.shape[0] + 1, dtype=int),
    "l": geom.positions[:, 0],
    "m": geom.positions[:, 1],
    "n": geom.positions[:, 2],
    "vx": geom.velocities.real[:, 0],
    "vy": geom.velocities.real[:, 1],
    "vz": geom.velocities.real[:, 2],
})
st.download_button(
    label="Download geometry+velocity CSV",
    data=geom_df.to_csv(index=False).encode("utf-8"),
    file_name="sphere_geom_vel.csv",
    mime="text/csv",
)

if st.checkbox("Show first 10 geometry points"):
    st.dataframe(geom_df.head(10))

st.write("## Step 2: Plane waves and pressure")

with st.spinner("Generating plane waves and computing pressure..."):
    pqr_set = generate_pqr(config)
    result = compute_pressure_dam_1_2(config, geom, pqr_set)

st.success("Computation complete.")

# Summary
st.subheader("Configuration summary")

col1, col2, col3 = st.columns(3)
with col1:
    st.write(f"N = {geom.positions.shape[0]}")
    st.write(f"Radius a = {float(radius):.6g}")
with col2:
    st.write(f"K = {config.K}")
    st.write(f"W = {config.W}")
with col3:
    st.write(f"λ = {config.lambda_:.6g}")
    st.write(f"Axis avoid = {config.axis_avoid_deg}°")

st.write(f"Plane-wave octant counts = {pqr_set.octant_counts}")
st.write(f"Mirror-pair count (should be 0) = {pqr_set.mirror_pair_count}")

delta_mag = float(result["delta_mag"])
delta_phase = float(result["delta_phase"])
st.write(f"Uniformity diagnostics: Δ|p|/mean(|p|) = {delta_mag:.6g}, Δphase = {delta_phase:.6g} rad")

# Plane waves download (DAM_1_2 format)
pqr = pqr_set.pqr
pw_df = pd.DataFrame({
    "idx": np.arange(1, pqr.shape[0] + 1, dtype=int),
    "p": pqr[:, 0],
    "q": pqr[:, 1],
    "r": pqr[:, 2],
    "Vx": result["Wx"],
    "Vy": result["Wy"],
    "Vz": result["Wz"],
})
st.download_button(
    label="Download plane waves CSV",
    data=pw_df.to_csv(index=False).encode("utf-8"),
    file_name="plane_waves.csv",
    mime="text/csv",
)

if st.checkbox("Show first 10 plane waves"):
    st.dataframe(pw_df.head(10))

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

st.download_button(
    label="Download full pressure CSV",
    data=out_df.to_csv(index=False).encode("utf-8"),
    file_name="pulsating_sphere_pressures.csv",
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
