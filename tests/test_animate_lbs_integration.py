import json
import os
import shutil
import tempfile
import unittest
import numpy as np

# We will test animate_lbs.save_obj_sequence
from data_process.mesh_animation.animate_lbs import save_obj_sequence


class TestAnimateLbsIntegration(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_save_obj_sequence_backwards_compatible(self):
        """Calling save_obj_sequence with positional (dirpath, vertices, faces) must succeed."""
        out_dir = os.path.join(self.test_dir, "legacy")
        verts = np.zeros((2, 3, 3))
        verts[0] = [[0, 0, 0], [1, 0, 0], [0, 1, 0]]
        verts[1] = [[0, 0, 0], [2, 0, 0], [0, 1, 0]]
        faces = [(0, 1, 2)]

        save_obj_sequence(out_dir, verts, faces)

        self.assertTrue(os.path.exists(os.path.join(out_dir, "0000.obj")))
        self.assertTrue(os.path.exists(os.path.join(out_dir, "0001.obj")))

        with open(os.path.join(out_dir, "0000.obj"), 'r') as f:
            content = f.read()
        self.assertIn("v ", content)
        self.assertIn("f ", content)

    def test_save_obj_sequence_with_uvs_and_normals(self):
        """save_obj_sequence should write vt and vn lines when UVs are provided."""
        out_dir = os.path.join(self.test_dir, "with_uvs")
        verts = np.zeros((1, 3, 3))
        verts[0] = [[0, 0, 0], [1, 0, 0], [0, 1, 0]]
        faces = [(0, 1, 2)]
        uvs = np.array([[0, 0], [1, 0], [0, 1]])
        face_uvs = [(0, 1, 2)]

        save_obj_sequence(out_dir, verts, faces, uvs=uvs, face_uv_indices=face_uvs)

        obj_file = os.path.join(out_dir, "0000.obj")
        with open(obj_file, 'r') as f:
            content = f.read()

        self.assertIn("vt ", content)
        self.assertIn("vn ", content)
        # Face corners should be v/vt/vn
        self.assertIn("1/1/", content)

    def test_save_obj_sequence_exports_collider_json(self):
        """save_obj_sequence with export_collision=True should create collider.json with AABB & sphere."""
        out_dir = os.path.join(self.test_dir, "with_collider")
        verts = np.zeros((2, 3, 3))
        verts[0] = [[0, 0, 0], [1, 0, 0], [0, 1, 0]]
        verts[1] = [[0, 0, 0], [1, 2, 0], [0, 1, 3]]
        faces = [(0, 1, 2)]

        save_obj_sequence(out_dir, verts, faces, export_collision=True)

        collider_path = os.path.join(out_dir, "collider.json")
        self.assertTrue(os.path.exists(collider_path))

        with open(collider_path, 'r') as f:
            data = json.load(f)

        self.assertIn("aabb", data)
        self.assertIn("sphere", data)
        self.assertIn("center", data["aabb"])
        self.assertIn("radius", data["sphere"])


if __name__ == '__main__':
    unittest.main()
