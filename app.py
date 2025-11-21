#!/usr/bin/env python3
"""
DAM 1.1 Phase II – single-file implementation.

Core tasks:
  - Read geometry & velocities from CSV.
  - Generate plane waves (p,q,r) for given W, K using spherical Fibonacci sampling.
  - Build g-matrix and column-norms Λ.
  - Build [V] diagonal from plane-wave direction cosines.
  - Compute pressure using Phase II operator:
        A_x = (1/N) * V * Λ^{-1} * (g^H v_x)
        A_y = ...
        A_z = ...
        p = g A_x + g A_y + g A_z
  - Write per-point real/imag, magnitude, phase of p to CSV.

Inputs (main run):
  - geometry CSV path
  - N: number of geometry points (used to sanity-check)
  - W: integerization multiplier
  - lambda_: wavelength
  - K: number of plane waves (default 3N)

Plane-wave cache:
  - .dam_cache/pqr_W{W}_K{K}.npz

Convenience:
  - precompute_default_pqrs() can build & cache p,q,r for K in [800,1600,2400,3200].
"""

from __future__ import annotations
import os
import math
import argparse
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List

import numpy as np
import pandas as pd


# -----------------------------
# Data models
# -----------------------------

@dataclass
class Phase2Config:
    geom_csv: str          # geometry+velocity CSV
    N: int                 # expected number of points
    W: float               # multiplier for integerizing plane-wave directions
    lambda_: float         # wavelength
    K: int                 # number of plane waves (columns of g)
    axis_avoid_deg: float = 20.0
    cache_dir: str = ".dam_cache"


@dataclass
class Geometry:
    positions: np.ndarray      # (N,3) [l,m,n]
    velocities: np.ndarray     # (N,3) complex [vx,vy,vz]
    meta: Dict[str, object]


@dataclass
class PlaneWaveSet:
    pqr: np.ndarray            # (K,3) int32
    octant_counts: Tuple[int, ...]


# -----------------------------
# Geometry & velocity reader
# -----------------------------

def read_geometry_csv(path: str) -> Geometry:
    """
    Read geometry + velocities from CSV.

    Expected columns:
      l, m, n,
      vx_real, vx_imag,
      vy_real, vy_imag,
      vz_real, vz_imag

    Returns:
      Geometry(
        positions: (N,3) float64,
        velocities: (N,3) complex128,
        meta: dict
      )
    """
    df = pd.read_csv(path)

    required = ["l", "m", "n",
                "vx_real", "vx_imag",
                "vy_real", "vy_imag",
                "vz_real", "vz_imag"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {path}: {missing}")

    positions = df[["l", "m", "n"]].to_numpy(dtype=float)
    vx = df["vx_real"].to_numpy(dtype=float) + 1j * df["vx_imag"].to_numpy(dtype=float)
    vy = df["vy_real"].to_numpy(dtype=float) + 1j * df["vy_imag"].to_numpy(dtype=float)
    vz = df["vz_real"].to_numpy(dtype=float) + 1j * df["vz_imag"].to_numpy(dtype=float)
    velocities = np.column_stack([vx, vy, vz]).astype(np.complex128)

    meta = {
        "source": os.path.basename(path),
        "units_length": "arbitrary",
        "units_velocity": "arbitrary",
    }

    return Geometry(positions=positions, velocities=velocities, meta=meta)


# -----------------------------
# Plane-wave utilities
# -----------------------------

def spherical_fibonacci(n: int) -> np.ndarray:
    """
    Generate n roughly uniformly-distributed unit vectors on S^2
    using a spherical Fibonacci (golden-angle) construction.

    Returns:
      (n,3) array of float64 unit vectors.
    """
    i = np.arange(n, dtype=np.float64)
    phi = (1 + 5 ** 0.5) / 2.0
    ga = 2.0 * np.pi * (1.0 - 1.0 / phi)
    z = 1.0 - 2.0 * (i + 0.5) / n
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    theta = ga * i
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    u = np.stack([x, y, z], axis=1)
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    return u


def within_axis_cone(u: np.ndarray, deg: float) -> np.ndarray:
    """
    Return boolean mask of directions to keep (outside ±deg about x,y,z axes).
    """
    if deg <= 0:
        return np.ones(len(u), dtype=bool)
    cos_thr = np.cos(np.deg2rad(deg))
    ux, uy, uz = np.abs(u[:, 0]), np.abs(u[:, 1]), np.abs(u[:, 2])
    bad = (ux >= cos_thr) | (uy >= cos_thr) | (uz >= cos_thr)
    return ~bad


def octant_index(v: np.ndarray) -> np.ndarray:
    """
    Map vectors to octant index 0..7 based on sign of components.

    Octant mapping:
      sign(x),sign(y),sign(z) → 3-bit index.
    """
    s = (v >= 0).astype(np.int32)
    return (s[:, 0] << 2) + (s[:, 1] << 1) + s[:, 2]


def dedupe_triplets(pqr: np.ndarray) -> np.ndarray:
    """
    Remove duplicate integer triplets, keeping first occurrence.
    """
    if len(pqr) == 0:
        return pqr
    a = np.ascontiguousarray(pqr, dtype=np.int32)
    b = a.view([("p", np.int32), ("q", np.int32), ("r", np.int32)])
    _, idx = np.unique(b, return_index=True)
    return a[np.sort(idx)]


def forbid_mirrors(pqr: np.ndarray) -> np.ndarray:
    """
    Remove mirror pairs (v and -v), keeping the first representative.
    """
    if len(pqr) == 0:
        return pqr
    a = np.ascontiguousarray(pqr, dtype=np.int32)
    seen = set()
    keep_idx: List[int] = []
    for i, (p, q, r) in enumerate(a):
        # Canonical representative up to sign:
        if (p, q, r) >= (-p, -q, -r):
            key = (p, q, r)
        else:
            key = (-p, -q, -r)
        if key in seen:
            continue
        seen.add(key)
        keep_idx.append(i)
    return a[np.array(keep_idx, dtype=np.int64)]


def plane_wave_cache_path(cache_dir: str, W: float, K: int) -> str:
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, f"pqr_W{W:g}_K{K}.npz")


def generate_pqr(config: Phase2Config) -> PlaneWaveSet:
    """
    Generate or load a set of plane-wave integer directions (p,q,r) of length K.

    Includes:
      - spherical Fibonacci base directions
      - ±axis_avoid_deg cone removal about x,y,z
      - zero-vector removal
      - duplicate removal
      - mirror-pair removal
      - simple octant balancing (target ≈ K/8 per octant)

    Cached on disk in .npz for reuse.
    """
    cache_path = plane_wave_cache_path(config.cache_dir, config.W, config.K)
    if os.path.exists(cache_path):
        data = np.load(cache_path)
        pqr = data["pqr"].astype(np.int32)
    else:
        # Base unit directions
        u = spherical_fibonacci(10000)
        # Axis avoidance
        u = u[within_axis_cone(u, config.axis_avoid_deg)]
        # Quantize: round(W*u)
        pqr = np.rint(config.W * u).astype(np.int32)
        # Remove zero vectors
        pqr = pqr[np.any(pqr != 0, axis=1)]
        # Remove duplicates
        pqr = dedupe_triplets(pqr)
        # Remove mirror pairs
        pqr = forbid_mirrors(pqr)

        # Octant balancing
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
        # Top-up sequentially if needed
        if len(selected) < config.K:
            for v in pqr:
                if len(selected) == config.K:
                    break
                selected.append(v)
        pqr = np.array(selected[:config.K], dtype=np.int32)

        # Save cache
        np.savez_compressed(cache_path, pqr=pqr)

    oct_counts = tuple(np.bincount(octant_index(pqr), minlength=8).tolist())
    return PlaneWaveSet(pqr=pqr, octant_counts=oct_counts)


def precompute_default_pqrs(W: float = 500.0,
                            Ks: Tuple[int, ...] = (800, 1600, 2400, 3200),
                            cache_dir: str = ".dam_cache") -> None:
    """
    Convenience function: generate and cache plane-wave sets for
    several K values (800,1600,2400,3200) at a given W.

    Call once if you want those ready to go.
    """
    for K in Ks:
        cfg = Phase2Config(
            geom_csv="",  # not used here
            N=0,
            W=W,
            lambda_=1.0,  # unused for pqr
            K=K,
            cache_dir=cache_dir
        )
        pw = generate_pqr(cfg)
        print(f"Cached pqr for W={W}, K={K}, octant_counts={pw.octant_counts}")


# -----------------------------
# Core Phase II operator
# -----------------------------

def build_g(positions: np.ndarray, pqr: np.ndarray, W: float, lambda_: float) -> np.ndarray:
    """
    Build g-matrix:

      g[i,k] = exp( j * 2π/(W λ) * ( l_i p_k + m_i q_k + n_i r_k ) )

    positions: (N,3) floats [l,m,n]
    pqr:       (K,3) ints   [p,q,r]
    """
    lmn = positions.astype(np.float64)
    pqr_f = pqr.astype(np.float64)
    scale = 2.0 * np.pi / (W * lambda_)
    dots = lmn @ pqr_f.T   # (N,K)
    phases = scale * dots
    g = np.exp(1j * phases)
    return g.astype(np.complex128)


def build_V_diagonal(pqr: np.ndarray) -> np.ndarray:
    """
    Build V_k diagonal entries from plane-wave directions.

    Per your description: V-functions are built from the
    inverse cosines of the vectors defining the plane waves.

    Implementation here:
      - For each (p,q,r) form a unit direction u = (ux,uy,uz).
      - Compute theta_x = arccos(ux), theta_y = arccos(uy), theta_z = arccos(uz).
      - Define scalar V_k = (theta_x + theta_y + theta_z)/3.

    If you prefer a different combination (e.g. using only one component),
    you can modify this function.
    """
    pqr_f = pqr.astype(np.float64)
    norms = np.linalg.norm(pqr_f, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    u = pqr_f / norms  # direction cosines

    ux = np.clip(u[:, 0], -1.0, 1.0)
    uy = np.clip(u[:, 1], -1.0, 1.0)
    uz = np.clip(u[:, 2], -1.0, 1.0)

    theta_x = np.arccos(ux)
    theta_y = np.arccos(uy)
    theta_z = np.arccos(uz)

    V = (theta_x + theta_y + theta_z) / 3.0  # shape (K,)
    return V.astype(np.float64)


def compute_pressure_phase2(config: Phase2Config,
                            geom: Geometry,
                            pqr_set: PlaneWaveSet) -> Dict[str, np.ndarray]:
    """
    Compute surface pressure using the DAM 1.1 Phase II operator:

      A_x = (1/N) * diag(V) * diag(Λ^{-1}) * (g^H v_x)
      A_y = ...
      A_z = ...
      p = g(A_x + A_y + A_z)

    Returns dict containing:
      - "p": complex128 (N,) pressures
      - "p_real", "p_imag", "p_mag", "p_phase": float64 (N,)
      - "Lambda": float64 (K,) diagonal of g^H g / N
      - "V": float64 (K,) diagonal entries
    """
    positions = geom.positions
    velocities = geom.velocities
    N = positions.shape[0]

    if N != config.N and config.N > 0:
        print(f"Warning: config.N={config.N} but file has N={N}; proceeding with N={N}.")
        config.N = N

    # Build g
    g = build_g(positions, pqr_set.pqr, config.W, config.lambda_)

    # Column norms Λ = diag(g^H g)/N
    Lambda = (g.conj().T @ g).diagonal().real / N
    Lambda_inv = 1.0 / np.maximum(Lambda, 1e-12)

    # Build V from plane-wave directions
    V = build_V_diagonal(pqr_set.pqr)  # shape (K,)

    # Pressure accumulation
    p_total = np.zeros(N, dtype=np.complex128)
    for comp in range(3):
        vcomp = velocities[:, comp]
        proj = g.conj().T @ vcomp                 # (K,)
        A = (1.0 / N) * V * Lambda_inv * proj     # diag(V) * Λ^{-1} * g^H v
        p_total += g @ A                          # g A

    p = p_total
    p_real = p.real
    p_imag = p.imag
    p_mag = np.abs(p)
    p_phase = np.angle(p)

    return {
        "p": p,
        "p_real": p_real,
        "p_imag": p_imag,
        "p_mag": p_mag,
        "p_phase": p_phase,
        "Lambda": Lambda,
        "V": V,
    }


# -----------------------------
# Output utilities
# -----------------------------

def write_pressure_csv(out_path: str,
                       geom: Geometry,
                       result: Dict[str, np.ndarray]) -> None:
    """
    Write per-point pressure data to CSV:

      l, m, n,
      p_real, p_imag, p_mag, p_phase
    """
    df = pd.DataFrame({
        "l": geom.positions[:, 0],
        "m": geom.positions[:, 1],
        "n": geom.positions[:, 2],
        "p_real": result["p_real"],
        "p_imag": result["p_imag"],
        "p_mag": result["p_mag"],
        "p_phase": result["p_phase"],
    })
    df.to_csv(out_path, index=False)
    print(f"Wrote pressure CSV: {out_path}")


# -----------------------------
# CLI
# -----------------------------

def main():
    parser = argparse.ArgumentParser(
        description="DAM 1.1 Phase II pressure calculation for a digitized radiator."
    )
    parser.add_argument("--geom", required=True,
                        help="Geometry + velocity CSV file.")
    parser.add_argument("--N", type=int, required=True,
                        help="Number of points (sanity check).")
    parser.add_argument("--W", type=float, required=True,
                        help="Multiplier W for plane-wave integerization.")
    parser.add_argument("--lambda", dest="lambda_", type=float, required=True,
                        help="Wavelength lambda.")
    parser.add_argument("--K", type=int, default=None,
                        help="Number of plane waves; defaults to 3*N.")
    parser.add_argument("--axis-avoid", type=float, default=20.0,
                        help="Axis avoidance half-angle in degrees (default 20).")
    parser.add_argument("--cache-dir", type=str, default=".dam_cache",
                        help="Directory to cache plane-wave sets.")
    parser.add_argument("--out-csv", type=str, default="dam11_phase2_pressures.csv",
                        help="Output CSV for pressure results.")
    parser.add_argument("--precompute-pqrs", action="store_true",
                        help="Precompute cached pqr sets for K=800,1600,2400,3200 and exit.")

    args = parser.parse_args()

    if args.precompute_pqrs:
        precompute_default_pqrs(W=args.W, cache_dir=args.cache_dir)
        return

    # Build config
    K = args.K if args.K is not None else 3 * args.N
    cfg = Phase2Config(
        geom_csv=args.geom,
        N=args.N,
        W=args.W,
        lambda_=args.lambda_,
        K=K,
        axis_avoid_deg=args.axis_avoid,
        cache_dir=args.cache_dir,
    )

    # Read geometry/velocities
    geom = read_geometry_csv(cfg.geom_csv)
    print(f"Read geometry from {cfg.geom_csv}: N={geom.positions.shape[0]}")

    # Generate or load plane waves
    pqr_set = generate_pqr(cfg)
    print(f"Using plane waves: K={cfg.K}, octant_counts={pqr_set.octant_counts}")

    # Compute pressure
    result = compute_pressure_phase2(cfg, geom, pqr_set)
    print(f"Lambda diag stats: min={result['Lambda'].min():.6g}, "
          f"max={result['Lambda'].max():.6g}, "
          f"mean={result['Lambda'].mean():.6g}")

    # Write outputs
    write_pressure_csv(args.out_csv, geom, result)

    # Optional: show simple global summary
    p = result["p"]
    print(f"Global mean p (Re, Im): ({p.real.mean():.6g}, {p.imag.mean():.6g})")
    print(f"Global mean |p|: {np.abs(p).mean():.6g}")


if __name__ == "__main__":
    main()
