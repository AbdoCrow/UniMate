import unittest
from unittest.mock import MagicMock

from data_process.utils.blender_rig import (
    calculate_decimate_ratio,
    decimate_mesh_for_game,
)


class MockModifier:
    def __init__(self, name, type):
        self.name = name
        self.type = type
        self.decimate_type = None
        self.ratio = 1.0


class MockModifiersCollection:
    def __init__(self):
        self._mods = []

    def __iter__(self):
        return iter(self._mods)

    def __len__(self):
        return len(self._mods)

    def new(self, name, type):
        mod = MockModifier(name, type)
        self._mods.append(mod)
        return mod


class MockMesh:
    def __init__(self, polygon_count):
        # List of dummy polygons of specified length
        self.polygons = [MagicMock() for _ in range(polygon_count)]


class MockMeshObject:
    def __init__(self, name, polygon_count):
        self.name = name
        self.type = 'MESH'
        self.data = MockMesh(polygon_count)
        self.modifiers = MockModifiersCollection()


class TestMeshDecimation(unittest.TestCase):
    # ── Pure calculation & validation tests ──────────────────────────────────
    def test_ratio_when_below_target(self):
        """When current triangles <= max_triangles, ratio must be 1.0 (no decimation)."""
        ratio = calculate_decimate_ratio(current_triangles=2000, max_triangles=4000)
        self.assertEqual(ratio, 1.0)

    def test_ratio_when_exactly_at_target(self):
        """When current triangles == max_triangles, ratio must be 1.0."""
        ratio = calculate_decimate_ratio(current_triangles=4000, max_triangles=4000)
        self.assertEqual(ratio, 1.0)

    def test_ratio_when_above_target(self):
        """When current triangles > max_triangles, ratio must be target / current."""
        ratio = calculate_decimate_ratio(current_triangles=8000, max_triangles=4000)
        self.assertAlmostEqual(ratio, 0.5, places=7)

    def test_ratio_custom_target(self):
        """Test with custom polygon count targets."""
        ratio = calculate_decimate_ratio(current_triangles=10000, max_triangles=2500)
        self.assertAlmostEqual(ratio, 0.25, places=7)

    def test_invalid_max_triangles_raises_error(self):
        """max_triangles <= 0 should raise ValueError."""
        with self.assertRaises(ValueError):
            calculate_decimate_ratio(current_triangles=1000, max_triangles=0)
        with self.assertRaises(ValueError):
            calculate_decimate_ratio(current_triangles=1000, max_triangles=-100)

    def test_invalid_current_triangles_raises_error(self):
        """Negative current_triangles should raise ValueError."""
        with self.assertRaises(ValueError):
            calculate_decimate_ratio(current_triangles=-5, max_triangles=4000)

    # ── Mock Blender object modifier integration tests ────────────────────────
    def test_no_modifier_created_when_mesh_below_target(self):
        """If mesh already has fewer polygons than target, do not add a modifier."""
        obj = MockMeshObject("LowPolyHero", 1500)
        mod = decimate_mesh_for_game(obj, max_triangles=4000)
        self.assertIsNone(mod)
        self.assertEqual(len(obj.modifiers), 0)

    def test_no_modifier_created_when_mesh_at_target(self):
        """If mesh has exactly the target count, do not add a modifier."""
        obj = MockMeshObject("ExactPolyHero", 4000)
        mod = decimate_mesh_for_game(obj, max_triangles=4000)
        self.assertIsNone(mod)
        self.assertEqual(len(obj.modifiers), 0)

    def test_modifier_created_when_mesh_above_target(self):
        """If mesh has 20000 polygons and target is 4000, create DECIMATE modifier with ratio 0.2."""
        obj = MockMeshObject("HighPolyMonster", 20000)
        mod = decimate_mesh_for_game(obj, max_triangles=4000)

        self.assertIsNotNone(mod)
        self.assertEqual(mod.type, 'DECIMATE')
        self.assertEqual(mod.decimate_type, 'COLLAPSE')
        self.assertAlmostEqual(mod.ratio, 0.2, places=6)
        self.assertEqual(len(obj.modifiers), 1)

    def test_default_target_is_4000(self):
        """When max_triangles is omitted, default target should be 4000."""
        obj = MockMeshObject("Monster", 8000)
        mod = decimate_mesh_for_game(obj)
        self.assertIsNotNone(mod)
        self.assertAlmostEqual(mod.ratio, 0.5, places=6)

    def test_existing_modifiers_preserved(self):
        """Existing modifiers on the mesh (e.g. Armature modifier) must not be removed or overwritten."""
        obj = MockMeshObject("RiggedCharacter", 10000)
        # Pre-existing Armature modifier
        arm_mod = obj.modifiers.new(name="Armature", type='ARMATURE')

        dec_mod = decimate_mesh_for_game(obj, max_triangles=5000)

        self.assertEqual(len(obj.modifiers), 2)
        self.assertIn(arm_mod, list(obj.modifiers))
        self.assertIn(dec_mod, list(obj.modifiers))


if __name__ == '__main__':
    unittest.main()
