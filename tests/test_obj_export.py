import os
import shutil
import tempfile
import unittest
import numpy as np

from data_process.utils.game_export import (
    export_game_obj_frame,
    export_game_obj_sequence,
    write_mtl_file,
    compute_vertex_normals,
)


class TestObjExport(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_basic_obj_export_with_uvs_and_normals(self):
        """Test exporting a single triangle with positions, UVs, and normals."""
        filepath = os.path.join(self.test_dir, "tri.obj")
        verts = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ])
        faces = [(0, 1, 2)]
        uvs = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ])
        normals = np.array([
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
        ])

        export_game_obj_frame(filepath, verts, faces, uvs=uvs, normals=normals)

        self.assertTrue(os.path.exists(filepath))
        with open(filepath, 'r') as f:
            lines = [line.strip() for line in f if line.strip() and not line.startswith('#')]

        v_lines = [l for l in lines if l.startswith('v ')]
        vt_lines = [l for l in lines if l.startswith('vt ')]
        vn_lines = [l for l in lines if l.startswith('vn ')]
        f_lines = [l for l in lines if l.startswith('f ')]

        self.assertEqual(len(v_lines), 3)
        self.assertEqual(len(vt_lines), 3)
        self.assertEqual(len(vn_lines), 3)
        self.assertEqual(len(f_lines), 1)

        # 1-based indexing check: f 1/1/1 2/2/2 3/3/3
        self.assertEqual(f_lines[0], "f 1/1/1 2/2/2 3/3/3")

    def test_uv_seam_shared_vertices_different_uvs(self):
        """Test UV seams where shared 3D vertices have distinct UV coordinates per face loop."""
        filepath = os.path.join(self.test_dir, "seam.obj")
        # 4 vertices forming two triangles sharing an edge (verts 1 and 2)
        verts = np.array([
            [0.0, 0.0, 0.0],  # 0
            [1.0, 0.0, 0.0],  # 1 (shared)
            [1.0, 1.0, 0.0],  # 2 (shared)
            [2.0, 0.0, 0.0],  # 3
        ])
        faces = [
            (0, 1, 2),
            (1, 3, 2),
        ]
        # 6 distinct UV coordinates (3 per face) to test loop mapping
        uvs = np.array([
            [0.0, 0.0],  # UV 0 for tri 0 corner 0
            [0.5, 0.0],  # UV 1 for tri 0 corner 1
            [0.5, 1.0],  # UV 2 for tri 0 corner 2
            [0.6, 0.0],  # UV 3 for tri 1 corner 0 (shared vert 1 on seam)
            [1.0, 0.0],  # UV 4 for tri 1 corner 1
            [0.6, 1.0],  # UV 5 for tri 1 corner 2 (shared vert 2 on seam)
        ])
        face_uv_indices = [
            (0, 1, 2),
            (3, 4, 5),
        ]

        export_game_obj_frame(filepath, verts, faces, uvs=uvs, face_uv_indices=face_uv_indices)

        with open(filepath, 'r') as f:
            lines = [l.strip() for l in f if l.strip()]

        f_lines = [l for l in lines if l.startswith('f ')]
        self.assertEqual(len(f_lines), 2)

        # In tri 1, vertex 1 uses UV 3 (1-based -> vert 2, UV 4)
        # and vertex 2 uses UV 5 (1-based -> vert 3, UV 6)
        # Note: normals are auto-computed if not provided
        self.assertTrue(f_lines[0].startswith("f 1/1/"))
        self.assertTrue(f_lines[1].startswith("f 2/4/"))

    def test_mesh_without_uvs_fallback(self):
        """A mesh with no UV layer should export valid f v//vn indices without crashing or invalid syntax."""
        filepath = os.path.join(self.test_dir, "no_uv.obj")
        verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
        faces = [(0, 1, 2)]
        normals = np.array([[0, 0, 1], [0, 0, 1], [0, 0, 1]])

        export_game_obj_frame(filepath, verts, faces, uvs=None, normals=normals)

        with open(filepath, 'r') as f:
            lines = [l.strip() for l in f if l.strip()]

        vt_lines = [l for l in lines if l.startswith('vt ')]
        self.assertEqual(len(vt_lines), 0)

        f_lines = [l for l in lines if l.startswith('f ')]
        self.assertEqual(f_lines[0], "f 1//1 2//2 3//3")

    def test_material_and_mtl_reference_written(self):
        """When mtl_filename and material_name are provided, mtllib and usemtl must be written."""
        filepath = os.path.join(self.test_dir, "mat.obj")
        verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
        faces = [(0, 1, 2)]

        export_game_obj_frame(
            filepath, verts, faces,
            mtl_filename="character.mtl",
            material_name="HeroMaterial",
        )

        with open(filepath, 'r') as f:
            content = f.read()

        self.assertIn("mtllib character.mtl", content)
        self.assertIn("usemtl HeroMaterial", content)

    def test_mtl_file_generation(self):
        """write_mtl_file should output valid Wavefront MTL syntax with diffuse texture map."""
        mtl_path = os.path.join(self.test_dir, "test.mtl")
        write_mtl_file(
            mtl_path,
            material_name="ZombieMat",
            diffuse_texture="zombie_albedo.png",
        )

        self.assertTrue(os.path.exists(mtl_path))
        with open(mtl_path, 'r') as f:
            content = f.read()

        self.assertIn("newmtl ZombieMat", content)
        self.assertIn("map_Kd zombie_albedo.png", content)

    def test_sequence_export_multiple_frames(self):
        """export_game_obj_sequence should create numbered OBJ files: 0000.obj, 0001.obj, ..."""
        seq_dir = os.path.join(self.test_dir, "anim_seq")
        frames = np.zeros((3, 3, 3))
        frames[0] = [[0, 0, 0], [1, 0, 0], [0, 1, 0]]
        frames[1] = [[0, 0, 0], [2, 0, 0], [0, 1, 0]]
        frames[2] = [[0, 0, 0], [3, 0, 0], [0, 1, 0]]
        faces = [(0, 1, 2)]

        export_game_obj_sequence(seq_dir, frames, faces, convert_to_opengl=True)

        self.assertTrue(os.path.exists(os.path.join(seq_dir, "0000.obj")))
        self.assertTrue(os.path.exists(os.path.join(seq_dir, "0001.obj")))
        self.assertTrue(os.path.exists(os.path.join(seq_dir, "0002.obj")))

        # Check coordinate conversion applied:
        # Vertex (1, 0, 0) in Blender -> (1, 0, 0) in OpenGL
        # Check that normals in each frame are recomputed
        with open(os.path.join(seq_dir, "0000.obj"), 'r') as f:
            content0 = f.read()
            self.assertIn("vn ", content0)

    def test_quad_face_obj_export(self):
        """Quad faces should export correctly as 4-corner faces."""
        filepath = os.path.join(self.test_dir, "quad.obj")
        verts = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]])
        faces = [(0, 1, 2, 3)]
        uvs = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])

        export_game_obj_frame(filepath, verts, faces, uvs=uvs)

        with open(filepath, 'r') as f:
            lines = [l.strip() for l in f if l.strip()]

        f_lines = [l for l in lines if l.startswith('f ')]
        self.assertEqual(len(f_lines), 1)
        corners = f_lines[0].split()[1:]
        self.assertEqual(len(corners), 4)

    def test_empty_mesh_handling(self):
        """Empty mesh (0 vertices, 0 faces) should produce a clean empty OBJ file without error."""
        filepath = os.path.join(self.test_dir, "empty.obj")
        export_game_obj_frame(filepath, np.zeros((0, 3)), [])
        self.assertTrue(os.path.exists(filepath))


if __name__ == '__main__':
    unittest.main()
