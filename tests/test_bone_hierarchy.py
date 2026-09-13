import json
import unittest
import numpy as np

from data_process.utils.game_export import (
    matrix_to_euler_zxy_degrees,
    euler_zxy_degrees_to_matrix,
    export_bone_hierarchy_jsonc,
)


class TestBoneHierarchyExport(unittest.TestCase):
    # ── Euler ZXY conversion tests ──────────────────────────────────────────
    def test_identity_matrix_to_euler(self):
        """Identity matrix should yield (0, 0, 0) degrees rotation."""
        eye = np.eye(3)
        euler = matrix_to_euler_zxy_degrees(eye)
        np.testing.assert_allclose(euler, [0.0, 0.0, 0.0], atol=1e-6)

    def test_known_pure_rotations(self):
        """Test pure single-axis rotations: 90 deg around X, Y, and Z."""
        # 90 deg around X: pitch
        # R_x(90 deg): [ [1, 0, 0], [0, 0, -1], [0, 1, 0] ]
        rx_90 = np.array([
            [1.0, 0.0,  0.0],
            [0.0, 0.0, -1.0],
            [0.0, 1.0,  0.0],
        ])
        euler_x = matrix_to_euler_zxy_degrees(rx_90)
        np.testing.assert_allclose(euler_x, [90.0, 0.0, 0.0], atol=1e-5)

        # 90 deg around Y: yaw
        # R_y(90 deg): [ [0, 0, 1], [0, 1, 0], [-1, 0, 0] ]
        ry_90 = np.array([
            [ 0.0, 0.0, 1.0],
            [ 0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
        ])
        euler_y = matrix_to_euler_zxy_degrees(ry_90)
        np.testing.assert_allclose(euler_y, [0.0, 90.0, 0.0], atol=1e-5)

        # 90 deg around Z: roll
        # R_z(90 deg): [ [0, -1, 0], [1, 0, 0], [0, 0, 1] ]
        rz_90 = np.array([
            [0.0, -1.0, 0.0],
            [1.0,  0.0, 0.0],
            [0.0,  0.0, 1.0],
        ])
        euler_z = matrix_to_euler_zxy_degrees(rz_90)
        np.testing.assert_allclose(euler_z, [0.0, 0.0, 90.0], atol=1e-5)

    def test_euler_zxy_roundtrip(self):
        """Test round-trip Euler -> Matrix -> Euler for arbitrary angles."""
        test_angles = [
            (15.0, 30.0, -45.0),
            (-20.0, 45.0, 60.0),
            (0.0, -35.0, 10.0),
            (80.0, -15.0, 0.0),
        ]
        for ex, ey, ez in test_angles:
            mat = euler_zxy_degrees_to_matrix(ex, ey, ez)
            recovered = matrix_to_euler_zxy_degrees(mat)
            np.testing.assert_allclose(recovered, [ex, ey, ez], atol=1e-5)

    # ── Hierarchy validation tests ──────────────────────────────────────────
    def test_duplicate_bone_names_raises_value_error(self):
        """Duplicate bone names must be rejected."""
        names = ["Spine", "Arm", "Arm"]
        parents = [-1, 0, 1]
        transforms = np.tile(np.eye(4), (3, 1, 1))

        with self.assertRaises(ValueError):
            export_bone_hierarchy_jsonc(names, parents, transforms)

    def test_cycle_in_hierarchy_raises_value_error(self):
        """Cyclic parent relationships must be detected and rejected."""
        names = ["A", "B", "C"]
        parents = [1, 2, 0]  # A -> B -> C -> A
        transforms = np.tile(np.eye(4), (3, 1, 1))

        with self.assertRaises(ValueError):
            export_bone_hierarchy_jsonc(names, parents, transforms)

    def test_invalid_parent_index_raises_value_error(self):
        """Parent index >= len(bone_names) must be rejected."""
        names = ["Root", "Child"]
        parents = [-1, 5]
        transforms = np.tile(np.eye(4), (2, 1, 1))

        with self.assertRaises(ValueError):
            export_bone_hierarchy_jsonc(names, parents, transforms)

    # ── Hierarchy JSONC export tests ────────────────────────────────────────
    def test_valid_hierarchy_jsonc_structure(self):
        """Verify export creates correct entities with parent strings and localTransform."""
        names = ["Root", "Spine", "Head"]
        parents = [-1, 0, 1]  # Root is parent of Spine, Spine is parent of Head

        # Transforms:
        # Root at (0, 0, 0)
        # Spine at (0, 1, 0) in OpenGL (which corresponds to (0, 0, 1) in Blender)
        # Head at (0, 2, 0) in OpenGL
        transforms = np.tile(np.eye(4), (3, 1, 1))
        # Root local pos: (0, 0, 0)
        transforms[1, :3, 3] = [0.0, 0.0, 1.0]  # Blender +Z=1
        transforms[2, :3, 3] = [0.0, 0.0, 1.0]  # Blender +Z=1

        entities = export_bone_hierarchy_jsonc(
            names, parents, transforms, convert_to_opengl=True
        )

        self.assertEqual(len(entities), 3)

        # Entity 0: Root
        self.assertEqual(entities[0]["name"], "Root")
        self.assertIsNone(entities[0]["parent"])
        np.testing.assert_allclose(entities[0]["localTransform"]["position"], [0, 0, 0], atol=1e-6)

        # Entity 1: Spine
        self.assertEqual(entities[1]["name"], "Spine")
        self.assertEqual(entities[1]["parent"], "Root")
        # Blender (0, 0, 1) -> OpenGL (0, 1, 0)
        np.testing.assert_allclose(entities[1]["localTransform"]["position"], [0, 1, 0], atol=1e-6)

        # Entity 2: Head
        self.assertEqual(entities[2]["name"], "Head")
        self.assertEqual(entities[2]["parent"], "Spine")
        np.testing.assert_allclose(entities[2]["localTransform"]["position"], [0, 1, 0], atol=1e-6)

        # Ensure valid JSON serialization
        serialized = json.dumps(entities)
        self.assertIn('"name": "Root"', serialized)
        self.assertIn('"parent": "Spine"', serialized)


if __name__ == '__main__':
    unittest.main()
