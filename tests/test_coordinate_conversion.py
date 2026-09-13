import unittest
import numpy as np

# This import will fail until we create the module
from data_process.utils.game_export import (
    blender_to_opengl_coords,
    BLENDER_TO_OPENGL_MATRIX,
)


class TestCoordinateConversion(unittest.TestCase):
    def test_canonical_axes_unit_vectors(self):
        """Test the three cardinal basis vectors:
        (1, 0, 0) -> (1, 0, 0)
        (0, 1, 0) -> (0, 0, -1)
        (0, 0, 1) -> (0, 1, 0)
        """
        x_axis = np.array([1.0, 0.0, 0.0])
        y_axis = np.array([0.0, 1.0, 0.0])
        z_axis = np.array([0.0, 0.0, 1.0])

        x_gl = blender_to_opengl_coords(x_axis)
        y_gl = blender_to_opengl_coords(y_axis)
        z_gl = blender_to_opengl_coords(z_axis)

        np.testing.assert_allclose(x_gl, [1.0, 0.0, 0.0], atol=1e-7)
        np.testing.assert_allclose(y_gl, [0.0, 0.0, -1.0], atol=1e-7)
        np.testing.assert_allclose(z_gl, [0.0, 1.0, 0.0], atol=1e-7)

    def test_matrix_representation(self):
        """Verify the 3x3 transformation matrix equals [[1, 0, 0], [0, 0, 1], [0, -1, 0]]."""
        expected_matrix = np.array([
            [1.0,  0.0,  0.0],
            [0.0,  0.0,  1.0],
            [0.0, -1.0,  0.0],
        ])
        np.testing.assert_allclose(BLENDER_TO_OPENGL_MATRIX, expected_matrix, atol=1e-7)
        # Determinant must be +1 (pure rotation, no reflection or hand-change)
        det = np.linalg.det(BLENDER_TO_OPENGL_MATRIX)
        self.assertAlmostEqual(det, 1.0, places=7)

    def test_single_vector_conversion(self):
        """Test conversion of an arbitrary 1D vector (3,)."""
        pt = np.array([2.5, -3.0, 4.2])
        # x_gl = 2.5, y_gl = 4.2, z_gl = -(-3.0) = 3.0
        expected = np.array([2.5, 4.2, 3.0])
        res = blender_to_opengl_coords(pt)
        np.testing.assert_allclose(res, expected, atol=1e-7)

    def test_multiple_vertices_2d_array(self):
        """Test conversion of (N, 3) vertex array."""
        pts = np.array([
            [1.0, 2.0, 3.0],
            [-4.0, -5.0, -6.0],
            [0.0, 0.0, 0.0],
            [10.5, -20.2, 30.1],
        ])
        expected = np.array([
            [1.0, 3.0, -2.0],
            [-4.0, -6.0, 5.0],
            [0.0, 0.0, 0.0],
            [10.5, 30.1, 20.2],
        ])
        res = blender_to_opengl_coords(pts)
        self.assertEqual(res.shape, pts.shape)
        np.testing.assert_allclose(res, expected, atol=1e-7)

    def test_multi_frame_vertices_3d_array(self):
        """Test conversion of animation vertices with shape (F, N, 3)."""
        frames = np.zeros((3, 2, 3))
        # Frame 0
        frames[0] = [[1, 2, 3], [4, 5, 6]]
        # Frame 1
        frames[1] = [[-1, -2, -3], [0, 1, 0]]
        # Frame 2
        frames[2] = [[0, 0, 1], [1, 0, 0]]

        res = blender_to_opengl_coords(frames)
        self.assertEqual(res.shape, (3, 2, 3))
        np.testing.assert_allclose(res[0, 0], [1, 3, -2], atol=1e-7)
        np.testing.assert_allclose(res[0, 1], [4, 6, -5], atol=1e-7)
        np.testing.assert_allclose(res[1, 0], [-1, -3, 2], atol=1e-7)
        np.testing.assert_allclose(res[1, 1], [0, 0, -1], atol=1e-7)
        np.testing.assert_allclose(res[2, 0], [0, 1, 0], atol=1e-7)
        np.testing.assert_allclose(res[2, 1], [1, 0, 0], atol=1e-7)

    def test_normals_remain_unit_length(self):
        """Test that surface normals transformed with the matrix remain unit length."""
        normals = np.array([
            [0.57735027, 0.57735027, 0.57735027],
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
        ])
        gl_normals = blender_to_opengl_coords(normals)
        norms = np.linalg.norm(gl_normals, axis=-1)
        np.testing.assert_allclose(norms, [1.0, 1.0, 1.0], atol=1e-6)

    def test_preserves_input_dtype(self):
        """Test that float32 input returns float32 output, float64 returns float64."""
        f32 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        res_f32 = blender_to_opengl_coords(f32)
        self.assertEqual(res_f32.dtype, np.float32)

        f64 = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        res_f64 = blender_to_opengl_coords(f64)
        self.assertEqual(res_f64.dtype, np.float64)

    def test_empty_array_handling(self):
        """Empty arrays with shape (0, 3) should return (0, 3) without error."""
        empty_pts = np.zeros((0, 3))
        res = blender_to_opengl_coords(empty_pts)
        self.assertEqual(res.shape, (0, 3))

    def test_invalid_shape_raises_value_error(self):
        """Shapes that do not have last dimension 3 must raise ValueError."""
        with self.assertRaises(ValueError):
            blender_to_opengl_coords(np.array([1.0, 2.0]))
        with self.assertRaises(ValueError):
            blender_to_opengl_coords(np.zeros((5, 4)))


if __name__ == '__main__':
    unittest.main()
