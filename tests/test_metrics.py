"""
tests/test_metrics.py
---------------------
Unit tests verifying the mathematical correctness of evaluation metric formulas
and asserting consistency with Team 16's test set evaluation results.
"""

import unittest
from scripts.metrics_summary import compute_all_metrics

class TestEvaluationMetricFormulas(unittest.TestCase):
    def test_perfect_model(self):
        """A model with 0 FP and 0 FN should achieve 100% across all metrics."""
        res = compute_all_metrics(tp=100, fn=0, fp=0, tn=100, roc_auc=1.0)
        self.assertAlmostEqual(res["Accuracy"][0], 1.0, places=4)
        self.assertAlmostEqual(res["Precision"][0], 1.0, places=4)
        self.assertAlmostEqual(res["Recall (Sensitivity)"][0], 1.0, places=4)
        self.assertAlmostEqual(res["Specificity"][0], 1.0, places=4)
        self.assertAlmostEqual(res["F1-Score"][0], 1.0, places=4)
        self.assertAlmostEqual(res["Balanced Accuracy"][0], 1.0, places=4)

    def test_imbalanced_trivial_model(self):
        """A model predicting all negative on imbalanced data has high accuracy but 0 recall."""
        # 90 TN, 10 FN, 0 TP, 0 FP
        res = compute_all_metrics(tp=0, fn=10, fp=0, tn=90, roc_auc=0.5)
        self.assertAlmostEqual(res["Accuracy"][0], 0.90, places=4)
        self.assertAlmostEqual(res["Recall (Sensitivity)"][0], 0.0, places=4)
        self.assertAlmostEqual(res["F1-Score"][0], 0.0, places=4)

    def test_team16_baseline_test_results(self):
        """Verifies Team 16's published confusion matrix values match reported metrics."""
        # Test counts: TP=2875, FN=393, FP=473, TN=1840 (Total=5,581)
        res = compute_all_metrics(tp=2875, fn=393, fp=473, tn=1840, roc_auc=0.9182)
        
        self.assertAlmostEqual(res["Accuracy"][0], 0.8448, places=4)
        self.assertAlmostEqual(res["Precision"][0], 0.8587, places=4)
        self.assertAlmostEqual(res["Recall (Sensitivity)"][0], 0.8797, places=4)
        self.assertAlmostEqual(res["F1-Score"][0], 0.8691, places=4)
        self.assertAlmostEqual(res["Specificity"][0], 0.7955, places=4)
        self.assertAlmostEqual(res["Balanced Accuracy"][0], 0.8376, places=4)

    def test_threshold_tradeoffs(self):
        """Verifies that shifting threshold from 0.3 to 0.7 increases precision while reducing recall."""
        # Threshold 0.3 values: TP=3091, FN=177, FP=784, TN=1529
        t3 = compute_all_metrics(tp=3091, fn=177, fp=784, tn=1529)
        # Threshold 0.7 values: TP=2559, FN=709, FP=258, TN=2055
        t7 = compute_all_metrics(tp=2559, fn=709, fp=258, tn=2055)

        self.assertGreater(t3["Recall (Sensitivity)"][0], t7["Recall (Sensitivity)"][0])
        self.assertLess(t3["Precision"][0], t7["Precision"][0])

if __name__ == "__main__":
    unittest.main()
