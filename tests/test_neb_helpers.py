import unittest

import numpy as np
from ase import Atoms

from tsase import neb


def make_atoms(positions):
    return Atoms(
        "Cu2",
        positions=positions,
        cell=np.diag([5.0, 5.0, 5.0]),
        pbc=[True, True, True],
    )


class NebHelperTests(unittest.TestCase):
    def test_compute_jacobian_regression_value(self):
        value = neb.compute_jacobian(100.0, 120.0, 4, weight=1.5)
        expected = ((110.0 / 4.0) ** (1.0 / 3.0)) * (4.0 ** 0.5) * 1.5
        self.assertTrue(np.isclose(value, expected))

    def test_generate_multi_point_path_preserves_ids_and_indices(self):
        start = make_atoms([[1.0, 2.0, 2.5], [4.0, 2.0, 2.5]])
        mid = make_atoms([[1.5, 2.5, 2.5], [3.5, 2.5, 2.5]])
        end = make_atoms([[2.0, 3.0, 2.5], [3.0, 3.0, 2.5]])

        path = neb.generate_multi_point_path([start, mid, end], [0, 2, 4], 5)

        self.assertEqual(len(path), 5)
        self.assertTrue(np.allclose(path[0].get_positions(), start.get_positions()))
        self.assertTrue(np.allclose(path[2].get_positions(), mid.get_positions()))
        self.assertTrue(np.allclose(path[4].get_positions(), end.get_positions()))
        for image in path:
            self.assertIn(neb.NEB_ATOM_ID_ARRAY, image.arrays)
            self.assertEqual(image.arrays[neb.NEB_ATOM_ID_ARRAY].tolist(), [0, 1])

    def test_spatial_map_reorders_permuted_structure(self):
        reference = make_atoms([[1.0, 2.0, 2.5], [4.0, 2.0, 2.5]])
        candidate = make_atoms([[4.0, 2.0, 2.5], [1.0, 2.0, 2.5]])
        neb.ensure_atom_ids(reference)

        reordered = neb.spatial_map(reference, candidate)

        self.assertTrue(np.allclose(reordered.get_positions(), reference.get_positions()))

    def test_generate_multi_point_path_preserves_caller_order_without_ids(self):
        start = make_atoms([[1.0, 2.0, 2.5], [4.0, 2.0, 2.5]])
        # This midpoint deliberately swaps the two Cu atoms. The path builder
        # should preserve the caller-supplied correspondence instead of
        # silently remapping it by nearest-neighbor geometry.
        midpoint = make_atoms([[3.7, 2.5, 2.5], [1.3, 2.5, 2.5]])
        end = make_atoms([[2.0, 3.0, 2.5], [3.0, 3.0, 2.5]])

        path = neb.generate_multi_point_path([start, midpoint, end], [0, 2, 4], 5)

        self.assertTrue(np.allclose(path[2].get_positions(), midpoint.get_positions()))


if __name__ == "__main__":
    unittest.main()
