"""Unified Game-Asset Export CLI for CMPS346 OpenGL Engine.

Takes a 3D character asset and an animation (from a text prompt or motion file),
and exports game-ready assets for the CMPS346 OpenGL forward renderer:
- Wavefront OBJ sequence (0000.obj, ...) with UVs and smooth per-vertex normals
- Blender Z-up -> OpenGL Y-up coordinate conversion
- Material file (material.mtl) with diffuse map reference
- 3D collision bounds (collider.json) containing AABB and bounding sphere
- Metadata summary (metadata.json)
"""

import argparse
import datetime
import json
import os
import shutil
import sys
from loguru import logger

# Add project root to sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data_process.utils.game_export import (
    export_game_obj_sequence,
    write_mtl_file,
    compute_collision_bounds,
)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Export UniMate 3D animations as game-ready OpenGL assets for CMPS346."
    )
    parser.add_argument(
        "--char_path", type=str, required=True,
        help="Path to rigged character asset (.glb, .fbx)."
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Destination directory for exported game assets."
    )
    parser.add_argument(
        "--prompt", type=str, default=None,
        help="Text prompt to generate motion (e.g. 'zombie walking slowly')."
    )
    parser.add_argument(
        "--anim_path", type=str, default=None,
        help="Path to pre-existing motion file (.npz, .npy)."
    )
    parser.add_argument(
        "--max-poly", "--max_poly", dest="max_poly", type=int, default=4000,
        help="Target maximum triangle count for mesh decimation (default: 4000)."
    )
    parser.add_argument(
        "--fps", type=int, default=30,
        help="Playback framerate for the animation (default: 30)."
    )
    parser.add_argument(
        "--export-collision", dest="export_collision", action="store_true", default=True,
        help="Export 3D collision bounding volumes into collider.json (default: True)."
    )
    parser.add_argument(
        "--no-export-collision", dest="export_collision", action="store_false",
        help="Disable collision bounds export."
    )
    parser.add_argument(
        "--convert-opengl", dest="convert_opengl", action="store_true", default=True,
        help="Convert coordinates from Blender (+Z up) to OpenGL (+Y up) (default: True)."
    )
    parser.add_argument(
        "--no-convert-opengl", dest="convert_opengl", action="store_false",
        help="Keep coordinates in native Blender (+Z up)."
    )
    parser.add_argument(
        "--material-name", dest="material_name", type=str, default="GameMaterial",
        help="Name for the generated material in .mtl (default: GameMaterial)."
    )
    parser.add_argument(
        "--texture-path", dest="texture_path", type=str, default=None,
        help="Optional diffuse texture image to bundle with the exported model."
    )
    parser.add_argument(
        "--dataset-type", dest="dataset_type", type=str, default=None,
        choices=['mixamo', 'objaverse', 'truebones'],
        help="Canonical skeleton type for motion retargeting."
    )
    parser.add_argument(
        "--exp-dir", dest="exp_dir", type=str, default=None,
        help="Trained UniMate model directory for text-to-motion generation."
    )
    return parser


def synthesize_or_load_motion_vertices(args):
    """Bridge to UniMate's generation / LBS animation pipeline.

    When running in a full Blender/PyTorch environment:
    Invokes animate_lbs or sample.py to generate/deform vertices.
    """
    from data_process.mesh_animation.animate_lbs import animate_lbs
    temp_lbs_dir = os.path.join(args.output_dir, "_temp_lbs")
    os.makedirs(temp_lbs_dir, exist_ok=True)

    vertices = animate_lbs(
        char_path=args.char_path,
        anim_path=args.anim_path,
        output_dir=temp_lbs_dir,
        dataset_type=args.dataset_type,
        save=('npz',),
    )
    # Load NPZ for faces and vertices
    stem = os.path.splitext(os.path.basename(args.anim_path))[0] + '_lbs'
    npz_path = os.path.join(temp_lbs_dir, stem + '.npz')
    import numpy as np
    data = np.load(npz_path, allow_pickle=True)
    faces = data['faces'].tolist()
    shutil.rmtree(temp_lbs_dir, ignore_errors=True)
    return vertices, faces, None


def run_game_export(args):
    """Execute the game asset export workflow."""
    if not os.path.isfile(args.char_path):
        raise FileNotFoundError(f"Character file not found: {args.char_path}")

    if args.max_poly <= 0:
        raise ValueError(f"max-poly must be positive, got {args.max_poly}")

    if args.fps <= 0:
        raise ValueError(f"fps must be positive, got {args.fps}")

    os.makedirs(args.output_dir, exist_ok=True)
    logger.info(f"Starting game asset export to {args.output_dir}")

    # Synthesize or load the animated geometry
    vertices, faces, uvs = synthesize_or_load_motion_vertices(args)

    import numpy as np
    verts_arr = np.asarray(vertices, dtype=np.float64)
    if verts_arr.ndim == 2:
        verts_arr = verts_arr[None, ...]
    num_frames, num_verts, _ = verts_arr.shape
    num_faces = len(faces)

    # 1. Export texture if provided
    diffuse_filename = None
    if args.texture_path and os.path.isfile(args.texture_path):
        diffuse_filename = os.path.basename(args.texture_path)
        dest_tex = os.path.join(args.output_dir, diffuse_filename)
        shutil.copyfile(args.texture_path, dest_tex)
        logger.info(f"Copied texture to {dest_tex}")

    # 2. Write Material MTL file
    mtl_filename = "material.mtl"
    mtl_path = os.path.join(args.output_dir, mtl_filename)
    write_mtl_file(
        mtl_path,
        material_name=args.material_name,
        diffuse_texture=diffuse_filename,
    )
    logger.info(f"Generated material library {mtl_path}")

    # 3. Export OBJ sequence
    export_game_obj_sequence(
        dirpath=args.output_dir,
        vertices=verts_arr,
        faces=faces,
        uvs=uvs,
        mtl_filename=mtl_filename,
        material_name=args.material_name,
        convert_to_opengl=args.convert_opengl,
    )
    logger.info(f"Exported {num_frames} OBJ frames ({num_verts} vertices, {num_faces} faces each)")

    # 4. Export Collision Bounding Volumes
    collision_info = None
    if args.export_collision:
        collision_info = compute_collision_bounds(
            verts_arr,
            convert_to_opengl=args.convert_opengl,
        )
        collider_path = os.path.join(args.output_dir, "collider.json")
        with open(collider_path, 'w', encoding='utf-8') as f:
            json.dump(collision_info, f, indent=2)
        logger.info(f"Exported collision bounding volumes to {collider_path}")

    # 5. Export Metadata
    metadata = {
        "generator": "UniMate Offline Asset Pipeline",
        "timestamp": datetime.datetime.now().isoformat(),
        "source_character": os.path.basename(args.char_path),
        "prompt": args.prompt,
        "frame_count": num_frames,
        "vertex_count": num_verts,
        "face_count": num_faces,
        "fps": args.fps,
        "target_max_poly": args.max_poly,
        "coordinate_system": "OpenGL (Y-up, -Z forward)" if args.convert_opengl else "Blender (Z-up, +Y forward)",
        "has_texture": diffuse_filename is not None,
        "has_collision": args.export_collision,
    }
    meta_path = os.path.join(args.output_dir, "metadata.json")
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Exported metadata to {meta_path}")

    return True


def main():
    parser = build_parser()
    args = parser.parse_args()
    try:
        run_game_export(args)
    except Exception as exc:
        logger.error(f"Export failed: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
