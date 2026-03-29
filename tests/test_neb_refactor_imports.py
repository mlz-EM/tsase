import unittest

from tsase import neb


class NebRefactorImportTests(unittest.TestCase):
    def test_public_facade_exports_core_symbols(self):
        self.assertIsNotNone(neb.ssneb)
        self.assertIsNotNone(neb.fire_ssneb)
        self.assertIsNotNone(neb.compute_jacobian)
        self.assertIsNotNone(neb.interpolate_path)
        self.assertEqual(neb.NEB_ATOM_ID_ARRAY, "neb_atom_id")
        self.assertIsNotNone(neb.EnthalpyWrapper)
        self.assertIsNotNone(neb.RunArtifacts)

    def test_legacy_import_shims_resolve(self):
        from tsase.neb.field import EnthalpyWrapper
        from tsase.neb.filtering import make_filter_adapter
        from tsase.neb.minimizer_ssneb import minimizer_ssneb
        from tsase.neb.run_artifacts import RunArtifacts
        from tsase.neb.ssneb import ssneb
        from tsase.neb.ssneb_utils import load_band_configuration_from_xyz
        from tsase.neb.stem_visualization import save_projected_neb_sequence

        self.assertIs(EnthalpyWrapper, neb.EnthalpyWrapper)
        self.assertTrue(callable(make_filter_adapter))
        self.assertIsNotNone(minimizer_ssneb)
        self.assertIs(RunArtifacts, neb.RunArtifacts)
        self.assertEqual(ssneb.__module__, "tsase.neb.core.band")
        self.assertIs(load_band_configuration_from_xyz, neb.load_band_configuration_from_xyz)
        self.assertIs(save_projected_neb_sequence, neb.save_projected_neb_sequence)


if __name__ == "__main__":
    unittest.main()
