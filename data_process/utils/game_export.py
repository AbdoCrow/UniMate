"""Game export utilities for CMPS346 OpenGL Engine.

Bridges UniMate's animations to clean, real-time OpenGL assets:
- Coordinate transformation (Blender Z-up -> OpenGL Y-up)
- Collision bounding volumes (AABB, Bounding Sphere)
- Smooth normal calculation for deformed meshes
- OBJ exporter with UVs, Normals, and Material MTL support
"""

import os
import numpy as np

# Transformation matrix from Blender (+Z up, +Y forward) to OpenGL (+Y up, -Z forward):
# x_gl =  x
# y_gl =  z
# z_gl = -y
BLENDER_TO_OPENGL_MATRIX = np.array([
    [1.0,  0.0,  0.0],
    [0.0,  0.0,  1.0],
    [0.0, -1.0,  0.0],
], dtype=np.float64)


def blender_to_opengl_coords(coords):
    """Convert coordinates or vectors from Blender (+Z up, +Y forward)
    to OpenGL (+Y up, -Z forward).

    Transformation:
        x_gl = x
        y_gl = z
        z_gl = -y

    Args:
        coords: numpy array of shape (..., 3).
                Supports (3,), (N, 3), or (F, N, 3).

    Returns:
        Converted coordinates of same shape and dtype.
    """
    arr = np.asarray(coords)
    if arr.ndim == 0 or arr.shape[-1] != 3:
        raise ValueError(f"Input must have last dimension 3, got shape {arr.shape}")

    x = arr[..., 0]
    y = arr[..., 1]
    z = arr[..., 2]

    res = np.stack([x, z, -y], axis=-1)
    if isinstance(coords, np.ndarray) and res.dtype != coords.dtype:
        res = res.astype(coords.dtype)
    return res


def compute_collision_bounds(vertices, convert_to_opengl=False):
    """Calculate axis-aligned bounding box (AABB) and bounding sphere
    enclosing all vertices across all animation frames.

    Args:
        vertices: numpy array of shape (..., 3). Supports (N, 3) or (F, N, 3).
        convert_to_opengl: if True, converts vertices to OpenGL coordinates
                           before calculating bounds.

    Returns:
        dict with native Python types ready for JSON export:
        {
            "aabb": {
                "center": [x, y, z],
                "half_size": [hx, hy, hz],
                "min": [min_x, min_y, min_z],
                "max": [max_x, max_y, max_z]
            },
            "sphere": {
                "center": [x, y, z],
                "radius": float
            }
        }
    """
    arr = np.asarray(vertices, dtype=np.float64)
    if arr.ndim == 0 or arr.shape[-1] != 3:
        raise ValueError(f"Input must have last dimension 3, got shape {arr.shape}")
    if arr.size == 0 or (arr.ndim > 1 and arr.shape[-2] == 0):
        raise ValueError("Cannot compute bounds on empty vertices array")
    if not np.all(np.isfinite(arr)):
        raise ValueError("Vertices array contains NaN or Inf values")

    if convert_to_opengl:
        arr = blender_to_opengl_coords(arr)

    flat = arr.reshape(-1, 3)

    min_pt = flat.min(axis=0)
    max_pt = flat.max(axis=0)
    center = (min_pt + max_pt) / 2.0
    half_size = (max_pt - min_pt) / 2.0

    dists = np.linalg.norm(flat - center, axis=-1)
    radius = float(dists.max())

    return {
        "aabb": {
            "center": [float(x) for x in center],
            "half_size": [float(x) for x in half_size],
            "min": [float(x) for x in min_pt],
            "max": [float(x) for x in max_pt],
        },
        "sphere": {
            "center": [float(x) for x in center],
            "radius": radius,
        }
    }

