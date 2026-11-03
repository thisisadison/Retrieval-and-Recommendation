import itertools
import unittest
import numpy as np
import pandas as pd
from project2_metrics import concordance_index, ranking_metrics, summarize_predictions, paired_bootstrap


class MetricTests(unittest.TestCase):
    def test_tie_expectation_matches_all_permutations(self):
        truth = np.array([0, 5, 5, 1, 0, 0])
        pred = np.array([3, 2, 2, 2, 1, 1])
        df = pd.DataFrame({"ligand_key": "a", "score": truth, "pred": pred})
        metric = ranking_metrics(df, ks=(1, 2, 3, 5)).iloc[0]
        ranks = []
        for middle in itertools.permutations([1, 2, 3]):
            for bottom in itertools.permutations([4, 5]):
                order = [0, *middle, *bottom]
                ranks.append(next(i + 1 for i, idx in enumerate(order) if truth[idx] == 5))
        self.assertAlmostEqual(metric["rank_of_best"], np.mean(ranks))
        for k in (1, 2, 3, 5):
            self.assertAlmostEqual(metric[f"hit@{k}"], np.mean(np.array(ranks) <= k))
        pd.testing.assert_frame_equal(ranking_metrics(df), ranking_metrics(df.iloc[::-1]))

    def test_constant_predictions_equal_random_ranking(self):
        df = pd.DataFrame({"ligand_key": "a", "score": [3, 3, 1, 0, 2, 1], "pred": 1})
        row = ranking_metrics(df).iloc[0]
        for k in (1, 5):
            self.assertAlmostEqual(row[f"hit@{k}"], row[f"random hit@{k}"])
        self.assertAlmostEqual(row["ci"], 0.5)
        self.assertAlmostEqual(row["rank_of_best"], 7 / 3)

    def test_ci_matches_pairwise_definition(self):
        y, pred = [1, 2, 2, 3, 5], [2, 1, 3, 3, 3]
        pairs = [(i, j) for i in range(len(y)) for j in range(i) if y[i] != y[j]]
        credit = [0.5 if pred[i] == pred[j] else float((y[i] - y[j]) * (pred[i] - pred[j]) > 0) for i, j in pairs]
        self.assertAlmostEqual(concordance_index(y, pred), np.mean(credit))

    def test_unrankable_ligands_are_counted_not_ranked(self):
        df = pd.DataFrame({"ligand_key": ["a", "a", "b", "b"], "score": [1, 1, 1, 2], "pred": [2, 0, 0, 1]})
        metrics = summarize_predictions(df)
        self.assertEqual(metrics["ranked ligands"], 1)
        self.assertEqual(metrics["excluded ligands"], 1)
        self.assertEqual(metrics["macro CI"], 1)
        all_equal = df.assign(score=1)
        self.assertTrue(np.isnan(summarize_predictions(all_equal)["Hit@1"]))

    def test_paired_cluster_bootstrap_preserves_exact_difference(self):
        df = pd.DataFrame({"ligand_key": ["a", "a", "b", "b", "b"],
                           "target_key": ["x", "y", "x", "y", "z"], "score": [1, 2, 0, 3, 1]})
        df["a_pred"], df["b_pred"] = df["score"] + 1, df["score"] + 2
        for cluster in ("ligand_key", "target_key"):
            result = paired_bootstrap(df, ["a_pred"], "b_pred", cluster=cluster, n_boot=100, seed=2)
            row = result[(result["model"] == "a_pred") & (result["metric"] == "MSE")].iloc[0]
            self.assertEqual(row["delta"], -3)
            self.assertEqual(row["delta_ci_low"], -3)
            self.assertEqual(row["delta_ci_high"], -3)


if __name__ == "__main__":
    unittest.main()
