import unittest
import numpy as np

from data_process.utils.game_export import compute_vertex_normals


class TestNormalsCalculation(unittest.TestCase):
    def test_single_triangle_face_normal(self):
        """A counter-clockwise triangle on XY plane at z=0 should have normal pointing in +Z [0, 0, 1]."""
        # Vertices: (0,0,0), (1,0,0), (0,1,0)
        verts = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ])
        faces = [(0, 1, 2)]

        normals = compute_vertex_normals(verts, faces)
        self.assertEqual(normals.shape, (3, 3))

        expected = np.array([
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
        ])
        np.testing.assert_allclose(normals, expected, atol=1e-6)

    def test_all_normals_are_unit_length(self):
        """All computed normals for non-degenerate vertices must be normalized (norm == 1.0)."""
        # Tetrahedron
        verts = np.array([
            [ 1.0,  1.0,  1.0],
            [-1.0, -1.0,  1.0],
            [-1.0,  1.0, -1.0],
            [ 1.0, -1.0, -1.0],
        ])
        faces = [
            (0, 1, 2),
            (0, 2, 3),
            (0, 3, 1),
            (1, 3, 2),
        ]
        normals = compute_vertex_normals(verts, faces)
        norms = np.linalg.norm(normals, axis=-1)
        np.testing.assert_allclose(norms, np.ones(4), atol=1e-6)

    def test_smooth_shared_vertex_accumulation(self):
        """Two adjacent triangles meeting at an edge should produce an average normal at the shared edge."""
        # Triangle 1 on XY plane (normal +Z)
        # Triangle 2 tilted 90 degrees onto XZ plane (normal +Y)
        verts = np.array([
            [0.0, 0.0, 0.0],  # shared vertex 0
            [1.0, 0.0, 0.0],  # shared vertex 1
            [0.0, 1.0, 0.0],  # tri 1 vertex 2 (y=1) -> normal = (0, 0, 1)
            [0.0, 0.0, 1.0],  # tri 2 vertex 3 (z=1) -> v1=(1,0,0), v3=(0,0,1)
        ])
        faces = [
            (0, 1, 2),  # (1,0,0) x (0,1,0) = (0,0,1)
            (0, 3, 1),  # (0,0,1) x (1,0,0) = (0,1,0)
        ]
        normals = compute_vertex_normals(verts, faces)

        # Unshared vertex 2 should point purely along +Z
        np.testing.assert_allclose(normals[2], [0.0, 0.0, 1.0], atol=1e-6)
        # Unshared vertex 3 should point purely along +Y
        np.testing.assert_allclose(normals[3], [0.0, 1.0, 0.0], atol=1e-6)

        # Shared vertices 0 and 1 receive (0,0,1) + (0,1,0) = (0, 1, 1), normalized -> (0, 1/sqrt(2), 1/sqrt(2))
        expected_shared = np.array([0.0, 1.0 / np.sqrt(2.0), 1.0 / np.sqrt(2.0)])
        np.testing.assert_allclose(normals[0], expected_shared, atol=1e-6)
        np.testing.assert_allclose(normals[1], expected_shared, atol=1e-6)

    def test_deformed_mesh_normals_change_per_frame(self):
        """When mesh vertices deform across animation frames, normals must update correspondingly."""
        # Frame 0: flat on XY plane (normal +Z)
        # Frame 1: flipped upside down (normal -Z)
        frames = np.zeros((2, 3, 3))
        frames[0] = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
        frames[1] = [
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],  # vertices 1 and 2 swapped in position, inverting winding
            [1.0, 0.0, 0.0],
        ]
        faces = [(0, 1, 2)]

        normals = compute_vertex_normals(frames, faces)
        self.assertEqual(normals.shape, (2, 3, 3))

        # Frame 0 normals must be +Z
        np.testing.assert_allclose(normals[0], [[0, 0, 1], [0, 0, 1], [0, 0, 1]], atol=1e-6)
        # Frame 1 normals must be -Z
        np.testing.assert_allclose(normals[1], [[0, 0, -1], [0, 0, -1], [0, 0, -1]], atol=1e-6)

    def test_degenerate_triangle_handling(self):
        """Degenerate triangles (zero area, collinear vertices) must not produce NaN or Inf."""
        # Degenerate triangle with all vertices at same position
        verts = np.array([
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
        ])
        faces = [(0, 1, 2)]

        normals = compute_vertex_normals(verts, faces, default_normal=(0.0, 1.0, 0.0))
        self.assertFalse(np.any(np.isnan(normals)))
        self.assertFalse(np.any(np.isinf(normals)))
        # Should fallback to default normal
        np.testing.assert_allclose(normals, [[0, 1, 0], [0, 1, 0], [0, 1, 0]], atol=1e-6)

    def test_collinear_degenerate_face(self):
        """Collinear vertices producing zero cross-product."""
        verts = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
        ])
        faces = [(0, 1, 2)]
        normals = compute_vertex_normals(verts, faces, default_normal=(0.0, 0.0, 1.0))
        self.assertFalse(np.any(np.isnan(normals)))
        np.testing.assert_allclose(normals, [[0, 0, 1], [0, 0, 1], [0, 0, 1]], atol=1e-6)

    def test_quad_faces_support(self):
        """Quad faces (4-tuples) should automatically triangulate and calculate smooth normals."""
        # Unit square on XY plane: 4 vertices, 1 quad face
        verts = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ])
        faces = [(0, 1, 2, 3)]

        normals = compute_vertex_normals(verts, faces)
        self.assertEqual(normals.shape, (4, 3))
        expected = np.array([
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
        ])
        np.testing.assert_allclose(normals, expected, atol=1e-6)

    def test_unreferenced_isolated_vertex(self):
        """An isolated vertex not in any face should safely receive the default normal."""
        verts = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [5.0, 5.0, 5.0],  # vertex 3 not in any face
        ])
        faces = [(0, 1, 2)]
        normals = compute_vertex_normals(verts, faces, default_normal=(0.0, 1.0, 0.0))
        np.testing.assert_allclose(normals[3], [0.0, 1.0, 0.0], atol=1e-6)


if __name__ == '__main__':
    unittest.main()
