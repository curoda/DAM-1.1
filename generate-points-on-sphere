import numpy as np
import pandas as pd

def spherical_fibonacci_points(N: int, radius: float) -> np.ndarray:
    """
    Generate N points approximately uniformly distributed on a sphere
    of given radius using the spherical Fibonacci method.
    Returns array of shape (N, 3) with columns [x, y, z].
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
    return pts, pts_unit  # positions, unit directions

def main():
    N = 80
    radius = 100.0

    # Geometry points and radial directions
    pts, dirs = spherical_fibonacci_points(N, radius)
    l, m, n = pts[:, 0], pts[:, 1], pts[:, 2]

    # Radial velocity of unity: v = unit radial vector
    vx = dirs[:, 0]
    vy = dirs[:, 1]
    vz = dirs[:, 2]

    # Imag parts are zero (purely real velocities)
    df = pd.DataFrame({
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

    out_name = "DAM_sphere80_geom_vel.csv"
    df.to_csv(out_name, index=False)
    print(f"Wrote {out_name} with shape {df.shape}")

if __name__ == "__main__":
    main()
