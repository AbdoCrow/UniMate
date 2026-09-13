import json
import unittest
import numpy as np

from data_process.utils.game_export import compute_collision_bounds, blender_to_opengl_coords


class TestCollisionBounds(unittest.TestCase):
    def test_single_vertex(self):
        """A single vertex should have identical min, max, and center, zero half-size, zero radius."""
        v = np.array([[[1.0, 2.0, 3.0]]])  # shape (1, 1, 3)
        bounds = compute_collision_bounds(v)

        self.assertIn("aabb", bounds)
        self.assertIn("sphere", bounds)

        aabb = bounds["aabb"]
        sphere = bounds["sphere"]

        np.testing.assert_allclose(aabb["min"], [1.0, 2.0, 3.0], atol=1e-7)
        np.testing.assert_allclose(aabb["max"], [1.0, 2.0, 3.0], atol=1e-7)
        np.testing.assert_allclose(aabb["center"], [1.0, 2.0, 3.0], atol=1e-7)
        np.testing.assert_allclose(aabb["half_size"], [0.0, 0.0, 0.0], atol=1e-7)

        np.testing.assert_allclose(sphere["center"], [1.0, 2.0, 3.0], atol=1e-7)
        self.assertAlmostEqual(sphere["radius"], 0.0, places=7)

    def test_single_frame_symmetric_box(self):
        """Test a unit cube [-1, 1]^3 centered at origin in a single frame."""
        # 8 corners of cube
        verts = np.array([
            [-1.0, -1.0, -1.0],
            [-1.0, -1.0,  1.0],
            [-1.0,  1.0, -1.0],
            [-1.0,  1.0,  1.0],
            [ 1.0, -1.0, -1.0],
            [ 1.0, -1.0,  1.0],
            [ 1.0,  1.0, -1.0],
            [ 1.0,  1.0,  1.0],
        ])
        bounds = compute_collision_bounds(verts)  # 2D input (N, 3)

        aabb = bounds["aabb"]
        sphere = bounds["sphere"]

        np.testing.assert_allclose(aabb["min"], [-1.0, -1.0, -1.0], atol=1e-7)
        np.testing.assert_allclose(aabb["max"], [1.0, 1.0, 1.0], atol=1e-7)
        np.testing.assert_allclose(aabb["center"], [0.0, 0.0, 0.0], atol=1e-7)
        np.testing.assert_allclose(aabb["half_size"], [1.0, 1.0, 1.0], atol=1e-7)

        np.testing.assert_allclose(sphere["center"], [0.0, 0.0, 0.0], atol=1e-7)
        # Distance from origin to (1, 1, 1) is sqrt(3) ~= 1.7320508
        self.assertAlmostEqual(sphere["radius"], np.sqrt(3.0), places=6)

    def test_asymmetric_geometry_with_negatives(self):
        """Test bounds on asymmetric shape with mixed negative and positive coordinates."""
        verts = np.array([
            [-5.0, 2.0, -3.0],
            [-1.0, 10.0, 1.0],
            [3.0, 4.0, -7.0],
        ])
        bounds = compute_collision_bounds(verts)

        expected_min = np.array([-5.0, 2.0, -7.0])
        expected_max = np.array([3.0, 10.0, 1.0])
        expected_center = (expected_min + expected_max) / 2.0  # [-1.0, 6.0, -3.0]
        expected_half = (expected_max - expected_min) / 2.0    # [4.0, 4.0, 4.0]

        np.testing.assert_allclose(bounds["aabb"]["min"], expected_min, atol=1e-7)
        np.testing.assert_allclose(bounds["aabb"]["max"], expected_max, atol=1e-7)
        np.testing.assert_allclose(bounds["aabb"]["center"], expected_center, atol=1e-7)
        np.testing.assert_allclose(bounds["aabb"]["half_size"], expected_half, atol=1e-7)

    def test_multiple_frames_animated_extension(self):
        """Test that an animated character whose arm extends farther in frame 2 expands the total bounds."""
        # 3 frames, 2 vertices each (body and hand)
        frames = np.zeros((3, 2, 3))

        # Frame 0: rest pose, hand at x=1.0
        frames[0] = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
        # Frame 1: hand reaches to x=2.5
        frames[1] = [[0.0, 0.0, 0.0], [2.5, 0.0, 0.0]]
        # Frame 2: hand reaches high to y=3.0
        frames[2] = [[0.0, 0.0, 0.0], [0.0, 3.0, 0.0]]

        bounds = compute_collision_bounds(frames)

        expected_min = [0.0, 0.0, 0.0]
        expected_max = [2.5, 3.0, 0.0]
        expected_center = [1.25, 1.5, 0.0]
        expected_half = [1.25, 1.5, 0.0]

        np.testing.assert_allclose(bounds["aabb"]["min"], expected_min, atol=1e-7)
        np.testing.assert_allclose(bounds["aabb"]["max"], expected_max, atol=1e-7)
        np.testing.assert_allclose(bounds["aabb"]["center"], expected_center, atol=1e-7)
        np.testing.assert_allclose(bounds["aabb"]["half_size"], expected_half, atol=1e-7)

        # Every single vertex in every frame must be enclosed by sphere
        sphere_center = np.array(bounds["sphere"]["center"])
        radius = bounds["sphere"]["radius"]
        dists = np.linalg.norm(frames - sphere_center, axis=-1)
        self.assertTrue(np.all(dists <= radius + 1e-6))

    def test_sphere_encloses_all_vertices_across_frames(self):
        """Statistical random test verifying the sphere encloses 100% of vertices."""
        np.random.seed(42)
        random_verts = np.random.uniform(-10.0, 10.0, size=(10, 50, 3))
        bounds = compute_collision_bounds(random_verts)

        center = np.array(bounds["sphere"]["center"])
        radius = bounds["sphere"]["radius"]
        dists = np.linalg.norm(random_verts - center, axis=-1)

        self.assertTrue(np.all(dists <= radius + 1e-6))
        self.assertAlmostEqual(dists.max(), radius, places=6)

    def test_json_serializability(self):
        """The returned dictionary must be directly serializable with json.dumps without NumPy type errors."""
        verts = np.array([[[1.0, 2.0, 3.0]]], dtype=np.float32)
        bounds = compute_collision_bounds(verts)
        try:
            serialized = json.dumps(bounds)
            deserialized = json.loads(serialized)
            self.assertEqual(deserialized["aabb"]["center"], [1.0, 2.0, 3.0])
        except TypeError as e:
            self.fail(f"compute_collision_bounds returned non-JSON-serializable types: {e}")

    def test_coordinate_conversion_interaction(self):
        """Test with convert_to_opengl=True: ensures bounds match the converted vertices."""
        # Vertex in Blender: x=1 (right), y=2 (forward), z=5 (up)
        # Converted to OpenGL: x_gl=1 (right), y_gl=5 (up), z_gl=-2 (back)
        blender_verts = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 5.0]])

        bounds_gl = compute_collision_bounds(blender_verts, convert_to_opengl=True)

        expected_gl_max = [1.0, 5.0, 0.0]
        expected_gl_min = [0.0, 0.0, -2.0]

        np.testing.assert_allclose(bounds_gl["aabb"]["min"], expected_gl_min, atol=1e-7)
        np.testing.assert_allclose(bounds_gl["aabb"]["max"], expected_gl_max, atol=1e-7)

    def test_empty_input_raises_value_error(self):
        """Empty arrays should raise ValueError."""
        with self.assertRaises(ValueError):
            compute_collision_bounds(np.zeros((0, 3)))
        with self.assertRaises(ValueError):
            compute_collision_bounds(np.zeros((5, 0, 3)))

    def test_invalid_dimensions_raises_value_error(self):
        """Arrays not ending in dimension 3 should raise ValueError."""
        with self.assertRaises(ValueError):
            compute_collision_bounds(np.zeros((10, 4)))

    def test_nan_or_inf_raises_value_error(self):
        """Arrays containing NaN or Inf must raise ValueError."""
        nan_verts = np.array([[1.0, 2.0, np.nan]])
        with self.assertRaises(ValueError):
            compute_collision_bounds(nan_verts)

        inf_verts = np.array([[1.0, np.inf, 3.0]])
        with self.assertRaises(ValueError):
            compute_collision_bounds(inf_verts)


if __name__ == '__main__':
    unittest.main()
