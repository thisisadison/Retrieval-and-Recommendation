import tempfile
from pathlib import Path
import unittest
import numpy as np
import pandas as pd
from project2_cf import split_pairs, build_index, fit_cf, predict, select_model, save_model, load_model, centred_cosine


def observations():
    return pd.DataFrame([{"ligand_key": str(i), "target_key": str(j),
                          "score": 5 + i / 10 + j / 100 + np.sin(i * j) / 10}
                         for i in range(10) for j in range(10)])


class CFTests(unittest.TestCase):
    def test_roundoff_constant_profiles_have_no_similarity(self):
        matrix = np.array([[11.7, 11.7, 11.7, np.nan],
                           [11.7, 11.7 + 1e-14, 11.7, np.nan],
                           [1.0, 2.0, 3.0, 4.0],
                           [2.0, 4.0, 6.0, 8.0]])
        similarity = centred_cosine(matrix)
        np.testing.assert_array_equal(similarity[:2], 0)
        self.assertAlmostEqual(similarity[2, 3], 1.0)

    def test_test_labels_do_not_change_selection(self):
        df, _ = split_pairs(observations())
        altered = df.copy()
        altered.loc[altered.split == "test", "score"] += 1000
        model, search = select_model(df, ks=(2, 3))
        other, other_search = select_model(altered, ks=(2, 3))
        pd.testing.assert_frame_equal(search, other_search)
        self.assertEqual(model["method"], other["method"])
        self.assertEqual(model["k"], other["k"])
        np.testing.assert_array_equal(model["E"], other["E"])

    def test_saved_model_reproduces_predictions(self):
        df, _ = split_pairs(observations())
        model = fit_cf(df[df.split == "train"], *build_index(df), method="item", k=3)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.npz"
            save_model(model, path, {"score_name": "test"}, seed=7)
            loaded = load_model(path)
            self.assertEqual(loaded["seed"], 7)
            np.testing.assert_array_equal(predict(model, df.ligand_key, df.target_key)[0],
                                          predict(loaded, df.ligand_key, df.target_key)[0])


if __name__ == "__main__":
    unittest.main()
