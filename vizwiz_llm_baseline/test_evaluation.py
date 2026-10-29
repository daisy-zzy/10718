"""Focused scientific correctness checks, independent of model outputs."""
import unittest
import numpy as np
from sklearn.metrics import average_precision_score, f1_score
from evaluate import BootstrapMetrics, binary_metrics, load_predictions
class EvaluationTests(unittest.TestCase):
    def test_bootstrap_ties_and_zero_weights(self):
        rng=np.random.default_rng(8)
        y=rng.integers(2,size=(137,9));p=np.round(rng.random((137,9)),1);w=rng.integers(4,size=137)
        actual=BootstrapMetrics(y,p).compute(w)
        np.testing.assert_allclose(actual[:9],[average_precision_score(y[:,j],p[:,j],sample_weight=w) for j in range(9)],rtol=1e-12)
        np.testing.assert_allclose(actual[9:],[f1_score(y[:,j],p[:,j]>.5,sample_weight=w,zero_division=0) for j in range(9)],rtol=1e-12)
    def test_ap_is_not_fixed_threshold_precision(self):
        m=binary_metrics([0,0,1,1],[.1,.8,.7,.9])
        self.assertEqual((m['TN'],m['FP'],m['FN'],m['TP']),(1,1,0,2))
        self.assertAlmostEqual(m['AP'],5/6);self.assertAlmostEqual(m['precision'],2/3)
    def test_missing_prediction_rejected(self):
        with self.assertRaises(ValueError):load_predictions([],['expected.jpg'])
if __name__=='__main__':unittest.main()
