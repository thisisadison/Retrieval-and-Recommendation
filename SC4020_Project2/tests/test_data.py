import unittest

import numpy as np
import pandas as pd

from project2_data import deduplicate, similarity_vs_profile, validate


class DataCleaningTests(unittest.TestCase):
    def test_davis_validity_and_units(self):
        values = pd.DataFrame({"score": [1.0, 10.0, 10000.0, 0.0, -1.0, np.inf, np.nan]})
        result = validate(values, "davis")
        np.testing.assert_allclose(result["score"], [9.0, 8.0, 5.0])
        self.assertEqual(len(result), 3)

    def test_kiba_zero_and_extreme_values_are_retained(self):
        result = validate(pd.DataFrame({"score": [0.0, 100.0, -2.0, np.nan]}), "kiba")
        np.testing.assert_allclose(result["score"], [0.0, 100.0, -2.0])

    def test_aggregation_preserves_provenance_and_averages_converted_labels(self):
        frame = pd.DataFrame({
            "ligand_id": ["L1", "L1", "L2"], "target_id": ["T1", "T2", "T1"],
            "smiles": ["C", "C", "C"], "sequence": ["ABC", "ABC", "ABC"],
            "score": [5.0, 7.0, 9.0], "at_screening_limit": [True, False, False],
        })
        pairs, ligands, targets = deduplicate(frame, ["at_screening_limit"])
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs.loc[0, "score"], 7.0)
        self.assertEqual(pairs.loc[0, "n_merged"], 3)
        self.assertAlmostEqual(pairs.loc[0, "frac_at_screening_limit"], 1 / 3)
        self.assertEqual(ligands.loc[0, "ligand_ids"], "L1|L2")
        self.assertEqual(targets.loc[0, "target_ids"], "T1|T2")

    def test_sequence_equality_is_used_instead_of_similarity_threshold(self):
        ids = ["T1", "T2", "T3"]
        similarity = pd.DataFrame([[1, .8, 1], [.8, 1, .5], [1, .5, 1]], index=ids, columns=ids)
        profiles = pd.DataFrame({"T1": [1, 2, 3], "T2": [1, 3, 2], "T3": [3, 2, 1]})
        result = similarity_vs_profile(similarity, profiles, {"T1": "AAA", "T2": "AAA", "T3": "BBB"}, 3)
        observed = set(zip(result["left_id"], result["right_id"]))
        self.assertNotIn(("T1", "T2"), observed)
        self.assertIn(("T1", "T3"), observed)

    def test_minimum_overlap_does_not_impute_missing_affinities(self):
        ids = ["A", "B"]
        similarity = pd.DataFrame([[1, .5], [.5, 1]], index=ids, columns=ids)
        profiles = pd.DataFrame({"A": [1, 2, 3, np.nan], "B": [np.nan, 3, 2, 1]})
        enough = similarity_vs_profile(similarity, profiles, {"A": "A", "B": "B"}, 2)
        self.assertEqual(enough.iloc[0]["overlap"], 2)
        too_few = similarity_vs_profile(similarity, profiles, {"A": "A", "B": "B"}, 3)
        self.assertTrue(too_few.empty)


if __name__ == "__main__":
    unittest.main()
