import json
import os
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

from scripts.export_for_game import build_parser, run_game_export


class TestExportForGameCLI(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.output_dir = os.path.join(self.test_dir, "output")
        # Dummy character file
        self.char_file = os.path.join(self.test_dir, "dummy_char.glb")
        with open(self.char_file, 'wb') as f:
            f.write(b"GLTF dummy content")

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_parser_defaults(self):
        """Verify default argument values."""
        parser = build_parser()
        args = parser.parse_args([
            "--char_path", self.char_file,
            "--output_dir", self.output_dir,
        ])
        self.assertEqual(args.char_path, self.char_file)
        self.assertEqual(args.output_dir, self.output_dir)
        self.assertEqual(args.max_poly, 4000)
        self.assertEqual(args.fps, 30)
        self.assertTrue(args.export_collision)
        self.assertTrue(args.convert_opengl)
        self.assertEqual(args.material_name, "GameMaterial")

    def test_parser_custom_arguments(self):
        """Verify CLI flags override defaults."""
        parser = build_parser()
        args = parser.parse_args([
            "--char_path", self.char_file,
            "--output_dir", self.output_dir,
            "--max-poly", "2500",
            "--fps", "60",
            "--no-export-collision",
            "--no-convert-opengl",
            "--material-name", "ZombieMat",
            "--prompt", "zombie walk",
        ])
        self.assertEqual(args.max_poly, 2500)
        self.assertEqual(args.fps, 60)
        self.assertFalse(args.export_collision)
        self.assertFalse(args.convert_opengl)
        self.assertEqual(args.material_name, "ZombieMat")
        self.assertEqual(args.prompt, "zombie walk")

    def test_missing_required_args_raises_system_exit(self):
        """Missing --char_path or --output_dir must exit with code 2."""
        parser = build_parser()
        with self.assertRaises(SystemExit):
            parser.parse_args(["--char_path", self.char_file])
        with self.assertRaises(SystemExit):
            parser.parse_args(["--output_dir", self.output_dir])

    def test_invalid_polygon_count(self):
        """max-poly <= 0 should raise ValueError during run_game_export."""
        parser = build_parser()
        args = parser.parse_args([
            "--char_path", self.char_file,
            "--output_dir", self.output_dir,
            "--max-poly", "0",
        ])
        with self.assertRaises(ValueError):
            run_game_export(args)

    def test_invalid_fps(self):
        """fps <= 0 should raise ValueError during run_game_export."""
        parser = build_parser()
        args = parser.parse_args([
            "--char_path", self.char_file,
            "--output_dir", self.output_dir,
            "--fps", "-10",
        ])
        with self.assertRaises(ValueError):
            run_game_export(args)

    def test_nonexistent_char_path(self):
        """Non-existent character file should raise FileNotFoundError."""
        parser = build_parser()
        args = parser.parse_args([
            "--char_path", os.path.join(self.test_dir, "nonexistent.glb"),
            "--output_dir", self.output_dir,
        ])
        with self.assertRaises(FileNotFoundError):
            run_game_export(args)

    def test_end_to_end_mocked_export(self):
        """Test complete export pipeline with mocked vertex synthesis."""
        parser = build_parser()
        args = parser.parse_args([
            "--char_path", self.char_file,
            "--output_dir", self.output_dir,
            "--max-poly", "3000",
            "--fps", "24",
            "--material-name", "TestHero",
        ])

        # Mock the heavy motion / LBS computation to return dummy vertices and faces
        mock_vertices = np.zeros((3, 4, 3))
        mock_vertices[0] = [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]]
        mock_vertices[1] = [[0, 0, 0], [1.1, 0, 0], [1.1, 1, 0], [0, 1, 0]]
        mock_vertices[2] = [[0, 0, 0], [1.2, 0, 0], [1.2, 1, 0], [0, 1, 0]]
        mock_faces = [(0, 1, 2), (0, 2, 3)]
        mock_uvs = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])

        with patch("scripts.export_for_game.synthesize_or_load_motion_vertices",
                   return_value=(mock_vertices, mock_faces, mock_uvs)):
            result = run_game_export(args)

        self.assertTrue(result)
        # Check generated files
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, "0000.obj")))
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, "0001.obj")))
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, "0002.obj")))
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, "material.mtl")))
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, "collider.json")))
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, "metadata.json")))

        # Verify metadata content
        with open(os.path.join(self.output_dir, "metadata.json"), 'r') as f:
            meta = json.load(f)

        self.assertEqual(meta["frame_count"], 3)
        self.assertEqual(meta["vertex_count"], 4)
        self.assertEqual(meta["face_count"], 2)
        self.assertEqual(meta["fps"], 24)
        self.assertEqual(meta["coordinate_system"], "OpenGL (Y-up, -Z forward)")

        # Verify collision data
        with open(os.path.join(self.output_dir, "collider.json"), 'r') as f:
            col = json.load(f)
        self.assertIn("aabb", col)
        self.assertIn("sphere", col)

    def test_collision_export_disabled(self):
        """When --no-export-collision is passed, collider.json must not be created."""
        parser = build_parser()
        args = parser.parse_args([
            "--char_path", self.char_file,
            "--output_dir", self.output_dir,
            "--no-export-collision",
        ])

        mock_vertices = np.zeros((1, 3, 3))
        mock_vertices[0] = [[0, 0, 0], [1, 0, 0], [0, 1, 0]]
        mock_faces = [(0, 1, 2)]

        with patch("scripts.export_for_game.synthesize_or_load_motion_vertices",
                   return_value=(mock_vertices, mock_faces, None)):
            run_game_export(args)

        self.assertFalse(os.path.exists(os.path.join(self.output_dir, "collider.json")))


if __name__ == '__main__':
    unittest.main()
