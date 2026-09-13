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


def compute_vertex_normals(vertices, faces, default_normal=(0.0, 1.0, 0.0)):
    """Calculate smooth area-weighted per-vertex normals for meshes or animation frames.

    Args:
        vertices: numpy array of shape (N, 3) or (F, N, 3).
        faces: iterable of face tuples (e.g. [(0, 1, 2), (0, 2, 3), ...] or quads).
        default_normal: fallback normal (tuple of 3 floats) for unreferenced or degenerate vertices.

    Returns:
        numpy array of shape (N, 3) or (F, N, 3) of unit-length normal vectors.
    """
    verts = np.asarray(vertices, dtype=np.float64)
    if verts.ndim == 0 or verts.shape[-1] != 3:
        raise ValueError(f"vertices must have last dimension 3, got shape {verts.shape}")

    is_single_frame = (verts.ndim == 2)
    if is_single_frame:
        frames = verts[None, ...]
    elif verts.ndim == 3:
        frames = verts
    else:
        raise ValueError(f"vertices must have 2 or 3 dimensions, got {verts.ndim}")

    num_frames, num_verts, _ = frames.shape
    def_norm = np.asarray(default_normal, dtype=np.float64)
    def_norm = def_norm / (np.linalg.norm(def_norm) + 1e-12)

    # Triangulate any n-gon faces into triangles
    triangles = []
    for face in faces:
        if len(face) < 3:
            continue
        elif len(face) == 3:
            triangles.append((face[0], face[1], face[2]))
        else:
            v0 = face[0]
            for k in range(1, len(face) - 1):
                triangles.append((v0, face[k], face[k + 1]))

    out_normals = np.zeros((num_frames, num_verts, 3), dtype=np.float64)

    if len(triangles) > 0 and num_verts > 0:
        tri_arr = np.array(triangles, dtype=np.int64)
        idx0 = tri_arr[:, 0]
        idx1 = tri_arr[:, 1]
        idx2 = tri_arr[:, 2]

        for f in range(num_frames):
            frame_verts = frames[f]
            e1 = frame_verts[idx1] - frame_verts[idx0]
            e2 = frame_verts[idx2] - frame_verts[idx0]
            fn = np.cross(e1, e2)

            np.add.at(out_normals[f], idx0, fn)
            np.add.at(out_normals[f], idx1, fn)
            np.add.at(out_normals[f], idx2, fn)

            norms = np.linalg.norm(out_normals[f], axis=-1, keepdims=True)
            valid = (norms[:, 0] > 1e-12)
            out_normals[f, valid] /= norms[valid]
            out_normals[f, ~valid] = def_norm
    else:
        out_normals[:] = def_norm

    if is_single_frame:
        return out_normals[0]
    return out_normals


def write_mtl_file(filepath, material_name, diffuse_texture=None,
                   ambient=(0.2, 0.2, 0.2), diffuse=(0.8, 0.8, 0.8),
                   specular=(0.5, 0.5, 0.5), shininess=32.0):
    """Write a standard Wavefront MTL material file compatible with CMPS346."""
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    lines = [
        f"# Material generated for CMPS346 OpenGL Engine",
        f"newmtl {material_name}",
        f"Ka {ambient[0]:.4f} {ambient[1]:.4f} {ambient[2]:.4f}",
        f"Kd {diffuse[0]:.4f} {diffuse[1]:.4f} {diffuse[2]:.4f}",
        f"Ks {specular[0]:.4f} {specular[1]:.4f} {specular[2]:.4f}",
        f"Ns {shininess:.2f}",
        "illum 2",
    ]
    if diffuse_texture:
        lines.append(f"map_Kd {diffuse_texture}")
    lines.append("")

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))


def export_game_obj_frame(filepath, vertices, faces, uvs=None, normals=None,
                          face_uv_indices=None, mtl_filename=None,
                          material_name=None, convert_to_opengl=False):
    """Write a single frame OBJ file with positions (v), UVs (vt), normals (vn),
    and faces (f v/vt/vn).

    Args:
        filepath: output .obj path.
        vertices: numpy array of shape (N, 3).
        faces: list/array of face tuples/lists (e.g. [(0, 1, 2), ...]).
        uvs: optional numpy array of shape (U, 2) or (N, 2).
        normals: optional numpy array of shape (N, 3). If None and faces exist,
                 normals will be auto-calculated using compute_vertex_normals.
        face_uv_indices: optional list of face UV tuples matching faces topology.
                         If None and uvs is provided, 1-to-1 vertex-to-UV mapping is assumed.
        mtl_filename: optional relative/absolute mtl file path to include in mtllib.
        material_name: optional material name to bind with usemtl.
        convert_to_opengl: if True, converts positions and normals from Blender Z-up
                           to OpenGL Y-up.
    """
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)

    verts = np.asarray(vertices, dtype=np.float64)
    if verts.ndim != 2 or (verts.shape[0] > 0 and verts.shape[1] != 3):
        raise ValueError(f"vertices must have shape (N, 3), got {verts.shape}")

    if convert_to_opengl and len(verts) > 0:
        verts = blender_to_opengl_coords(verts)

    # Compute normals if not provided and geometry exists
    if normals is None and len(verts) > 0 and len(faces) > 0:
        norms = compute_vertex_normals(verts, faces)
    elif normals is not None and len(normals) > 0:
        norms = np.asarray(normals, dtype=np.float64)
        if convert_to_opengl:
            norms = blender_to_opengl_coords(norms)
    else:
        norms = None

    has_uvs = (uvs is not None and len(uvs) > 0)
    has_norms = (norms is not None and len(norms) > 0)

    lines = [
        f"# Exported for CMPS346 OpenGL Engine",
    ]

    if mtl_filename:
        lines.append(f"mtllib {mtl_filename}")
    if material_name:
        lines.append(f"usemtl {material_name}")

    # Write vertices
    for x, y, z in verts:
        lines.append(f"v {x:.6f} {y:.6f} {z:.6f}")

    # Write texture coordinates
    if has_uvs:
        for u, v in uvs:
            lines.append(f"vt {u:.6f} {v:.6f}")

    # Write normals
    if has_norms:
        for nx, ny, nz in norms:
            lines.append(f"vn {nx:.6f} {ny:.6f} {nz:.6f}")

    # Write faces (1-based indexing in OBJ)
    for f_idx, poly in enumerate(faces):
        corners = []
        poly_uvs = face_uv_indices[f_idx] if (face_uv_indices is not None and f_idx < len(face_uv_indices)) else None

        for c_idx, v_idx in enumerate(poly):
            v_num = v_idx + 1
            vt_num = None
            if has_uvs:
                if poly_uvs is not None and c_idx < len(poly_uvs):
                    vt_num = poly_uvs[c_idx] + 1
                elif v_idx < len(uvs):
                    vt_num = v_idx + 1

            vn_num = (v_idx + 1) if has_norms else None

            if vt_num is not None and vn_num is not None:
                corners.append(f"{v_num}/{vt_num}/{vn_num}")
            elif vt_num is not None:
                corners.append(f"{v_num}/{vt_num}")
            elif vn_num is not None:
                corners.append(f"{v_num}//{vn_num}")
            else:
                corners.append(f"{v_num}")

        lines.append("f " + " ".join(corners))

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines) + "\n")


def export_game_obj_sequence(dirpath, vertices, faces, uvs=None, normals=None,
                             face_uv_indices=None, mtl_filename=None,
                             material_name=None, convert_to_opengl=True):
    """Write a sequence of OBJ files (0000.obj, 0001.obj, ...) for an animation.

    Args:
        dirpath: directory to store the OBJ sequence.
        vertices: numpy array of shape (F, N, 3).
        faces: list of face polygons.
        uvs: optional UV array (U, 2).
        normals: optional normals of shape (F, N, 3). If None, calculated per frame.
        face_uv_indices: optional face UV loop indices.
        mtl_filename: optional material library filename.
        material_name: optional material name.
        convert_to_opengl: if True, converts positions and normals to OpenGL Y-up.
    """
    os.makedirs(dirpath, exist_ok=True)
    verts = np.asarray(vertices, dtype=np.float64)
    if verts.ndim != 3:
        raise ValueError(f"vertices sequence must have shape (F, N, 3), got {verts.shape}")

    num_frames = verts.shape[0]
    for f in range(num_frames):
        frame_path = os.path.join(dirpath, f"{f:04d}.obj")
        frame_norms = normals[f] if normals is not None else None
        export_game_obj_frame(
            frame_path,
            verts[f],
            faces,
            uvs=uvs,
            normals=frame_norms,
            face_uv_indices=face_uv_indices,
            mtl_filename=mtl_filename,
            material_name=material_name,
            convert_to_opengl=convert_to_opengl,
        )



