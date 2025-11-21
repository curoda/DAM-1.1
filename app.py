# app.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from dam11_phase2_core import (
    Phase2Config,
    geometry_from_dataframe,
    generate_pqr,
    compute_pressure_phase2,
)


st.set_page_config(
    page_title="DAM 1.1 Phase II – Pulsating Sphere",
    layout="wide",
)

st.title("DAM 1.1 Phase II – Pulsating Sphere Surface Pressure")

st.markdown(
    """
This app generates a **pulsating sphere** test case and computes surface
pressure using the DAM 1.1 Phase II operator.

Instead of uploading geometry, the app constructs it for you:

- Points on a sphere of radius `R` using spherical Fibonacci sampling  
- Radial unit velocity at each point (pulsating sphere)
"""
)


# -----------------------------
# Spherical Fibonacci geometry
# -----------------------------

def spherical_fibonacci_points(N: int, radius: float):
    """
    Generate N points on a sphere of given radius using the spherical
    Fibonacci method, plus associated unit radial directions.

    Returns:
      pts  : (N,3) positions [l,m,n]
      dirs : (N,3) unit radial vectors
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
    # Normalize to be safe
    pts_unit /= np.linalg.norm(pts_unit, axis=1, keepdims=True)

    pts = radius * pts_unit
    return pts, pts_unit


# -----------------------------
# Sidebar controls
# -----------------------------

st.sidebar.header("Sphere & Phase II Parameters")

# Geometry
N = st.sidebar.number_input(
    "N (number of surface points)",
    min_value=4,
    value=80,
    step=1,
    help="Number of points on the spherical radiator surface.",
)

radius = st.sidebar.number_input(
    "Sphere radius",
    min_value=1.0,
    value=100.0,
    step=1.0,
    help="Radius of the pulsating sphere (same units as positions).",
)

# Plane-wave parameters
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
        value=2.0 * np.pi * radius,  # consistent with ka≈1 for guidance
        step=1.0,
        format="%.6f",
    )
else:
    # Enforce ka = 1 with radius
    # ka = k a = 1 → k = 1/a → λ = 2π / k = 2π a
    k = 1.0 / radius
    lambda_val = 2.0 * np.pi / k
    st.sidebar.markdown(
        f"Computed from ka = 1 and radius a = {radius:.3f}: "
        f"λ ≈ {lambda_val:.3f}"
    )

# Candidate K values
default_K = 3 * N
predefined_Ks = [800, 1600, 2400, 3200, default_K]
# ensure uniqueness
predefined_Ks = sorted(set(predefined_Ks))

K = st.sidebar.selectbox(
    "K (number of plane waves)",
    options=predefined_Ks,
    index=predefined_Ks.index(default_K) if default_K in predefined_Ks else len(predefined_Ks) - 1,
    help="3N is a common default; you can also pick 800,1600,2400,3200.",
)

axis_avoid = st.sidebar.number_input(
    "Axis avoid angle (deg)",
    min_value=0.0,
    max_value=89.9,
    value=20.0,
    step=1.0,
    help="Cone half-angle about coordinate axes to avoid plane waves too close to axes.",
)

run_button = st.sidebar.button("Run Phase II computation")


# -----------------------------
# Generate geometry & show preview
# -----------------------------

# Generate geometry and radial velocities now so user can see them
pts, dirs = spherical_fibonacci_points(N, radius)
l, m, n = pts[:, 0], pts[:, 1], pts[:, 2]
vx, vy, vz = dirs[:, 0], dirs[:, 1], dirs[:, 2]

df_geom = pd.DataFrame({
    "l": l,
    "m": m,
    "n": n,
    "vx_real": vx,
    "vx_imag": np.zeros_like(vx),
    "vy_real": vy,
    "vy_imag": np.zeros_like(vy),
    "vz_real": vz,
    "vz_imag": np.zeros_like(vz),
})

st.subheader("Generated geometry and radial velocity (preview)")
st.write(f"Sphere radius = {radius}, N = {N}")
st.dataframe(df_geom.head())

# Optional: simple 3D scatter of the geometry
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


# -----------------------------
# Run Phase II
# -----------------------------

if not run_button:
    st.stop()

st.info("Running DAM 1.1 Phase II with generated pulsating-sphere data...")

# Build Geometry from DataFrame
try:
    geom = geometry_from_dataframe(df_geom)
except Exception as e:
    st.error(f"Internal geometry build error: {e}")
    st.stop()

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


# -----------------------------
# Summary
# -----------------------------

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


# -----------------------------
# Pressure table and download
# -----------------------------

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


# -----------------------------
# Visualization of |p|
# -----------------------------

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
