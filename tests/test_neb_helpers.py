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
    def _path_segment_lengths(self, images):
        snapshots = [image.copy() for image in images]
        jacobian = neb.compute_jacobian(
            snapshots[0].get_volume(),
            snapshots[-1].get_volume(),
            len(snapshots[0]),
        )
        for image in snapshots:
            neb.initialize_image_properties(image, jacobian)
        lengths = []
        for left, right in zip(snapshots, snapshots[1:]):
            delta = neb.image_distance_vector(right, left)
            lengths.append(float(np.linalg.norm(delta)))
        return lengths

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

    def test_uniform_remesh_preserves_endpoints_ids_and_field_charges(self):
        start = make_atoms([[1.0, 2.0, 2.5], [4.0, 2.0, 2.5]])
        mid = make_atoms([[1.4, 2.4, 2.5], [3.6, 2.4, 2.5]])
        end = make_atoms([[2.0, 3.0, 2.5], [3.0, 3.0, 2.5]])
        path = neb.generate_multi_point_path([start, mid, end], [0, 2, 4], 5)
        for image in path:
            neb.attach_field_charges(image, [1.0, -1.0])

        remeshed = neb.uniform_remesh(path, num_images=7)

        self.assertEqual(len(remeshed), 7)
        self.assertTrue(np.allclose(remeshed[0].get_positions(), path[0].get_positions()))
        self.assertTrue(np.allclose(remeshed[-1].get_positions(), path[-1].get_positions()))
        for image in remeshed:
            self.assertEqual(image.arrays[neb.NEB_ATOM_ID_ARRAY].tolist(), [0, 1])
            self.assertTrue(np.allclose(image.arrays["field_charges"], [1.0, -1.0]))

    def test_uniform_remesh_supports_ratio_and_produces_uniform_metric_spacing(self):
        start = make_atoms([[1.0, 2.0, 2.5], [4.0, 2.0, 2.5]])
        end = make_atoms([[2.0, 3.0, 2.5], [3.0, 3.0, 2.5]])
        path = neb.interpolate_path(start, end, 5)

        remeshed = neb.uniform_remesh(path, upsample_ratio=1.5)

        self.assertEqual(len(remeshed), 7)
        lengths = self._path_segment_lengths(remeshed)
        self.assertLess(max(lengths) - min(lengths), 1.0e-10)


if __name__ == "__main__":
    unittest.main()
