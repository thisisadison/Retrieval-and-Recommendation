"""Behavioral checks for the cold-target extension and its inference boundary."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from project2_cold import (cold_target_split, confidence_weights, fit, neighbour_components,
                           predict, predict_new_target, save_model, load_model, sequence_similarity)


class ColdTargetTests(unittest.TestCase):
    def setUp(self):
        self.train = pd.DataFrame({"ligand_key": ["a", "a", "b", "b"],
                                   "target_key": ["x", "y", "x", "z"],
                                   "score": [2.0, 4.0, 1.0, 5.0]})
        self.model = fit(self.train)

    def test_nearest_similarity_uses_ligand_observations_only(self):
        similarity = pd.DataFrame([[0.1, 0.2, 0.99]], index=["new"], columns=["x", "y", "z"])
        queries = pd.DataFrame({"ligand_key": ["a", "b"], "target_key": ["new", "new"]})
        base, raw, used, nearest = neighbour_components(self.model, queries, similarity, k=1)
        np.testing.assert_allclose(nearest, [0.2, 0.99])
        np.testing.assert_allclose(raw, [4.0, 5.0])
        np.testing.assert_array_equal(used, [1, 1])

    def test_zero_similarity_falls_back(self):
        sim = pd.DataFrame([[0.0, 0.0, 0.0]], index=["new"], columns=["x", "y", "z"])
        queries = pd.DataFrame({"ligand_key": ["a"], "target_key": ["new"]})
        pred, used = predict(self.model, queries, S=sim, method="similarity_shrinkage", low=0.1, high=0.5)
        np.testing.assert_allclose(pred, [3.0])
        np.testing.assert_array_equal(used, [0])

    def test_shrinkage_interpolates_and_validates(self):
        np.testing.assert_allclose(confidence_weights(np.array([0, 0.1, 0.3, 0.5, 1]),
                                                     "similarity_shrinkage", low=0.1, high=0.5),
                                   [0, 0, 0.5, 1, 1])
        with self.assertRaises(ValueError):
            confidence_weights([0.5], "similarity_shrinkage", low=0.5, high=0.5)

    def test_unknown_ligand_fails_clearly(self):
        sim = pd.DataFrame([[1.0]], index=["new"], columns=["x"])
        queries = pd.DataFrame({"ligand_key": ["unknown"], "target_key": ["new"]})
        with self.assertRaisesRegex(ValueError, "Unsupported ligand"):
            predict(self.model, queries, S=sim)

    def test_saved_model_new_sequence_matches_batch_prediction(self):
        sequences = pd.Series({"x": "AAACCCDDD", "y": "AAACCCEEE", "z": "GGGHHHIII"})
        query_sequence = "AAACCCFFF"
        all_sequences = pd.DataFrame({"sequence": pd.concat([sequences, pd.Series({"new": query_sequence})])})
        similarity = sequence_similarity(all_sequences)
        query = pd.DataFrame({"ligand_key": ["a", "b"], "target_key": ["new", "new"]})
        params = dict(method="similarity_shrinkage", k=2, alpha=1.0, low=0.0, high=0.5)
        before, _ = predict(self.model, query, S=similarity, **params)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "cold.npz"
            save_model(self.model, path, sequences, params, "davis", 42)
            restored = load_model(path)
            after, _ = predict(restored, query, S=similarity, **params)
            direct = predict_new_target(path, query_sequence, ["a", "b"])
            np.testing.assert_allclose(after, before, atol=1e-12)
            np.testing.assert_allclose(direct["prediction"], before, atol=1e-12)
            with self.assertRaisesRegex(ValueError, "amino-acid"):
                predict_new_target(path, "???", ["a"])

    def test_predictions_ignore_query_labels(self):
        similarity = pd.DataFrame([[0.1, 0.2, 0.3]], index=["new"], columns=["x", "y", "z"])
        query = pd.DataFrame({"ligand_key": ["a"], "target_key": ["new"], "score": [0.0]})
        before, _ = predict(self.model, query, S=similarity)
        after, _ = predict(self.model, query.assign(score=999.0), S=similarity)
        np.testing.assert_array_equal(before, after)


if __name__ == "__main__":
    unittest.main()
