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
