import unittest
import numpy as np
import pandas as pd

from decision_model import ValidationError, evaluate
from sample_data import sample_criteria, sample_matrix


class DecisionModelTests(unittest.TestCase):
    def setUp(self):
        self.matrix = sample_matrix()
        self.criteria = sample_criteria()

    def test_independent_two_option_hand_calculation(self):
        # Benefit: [3,4]/5; cost: [4,3]/5; each weight .5.
        # A is worst on both: D+ = sqrt(.1^2+.1^2), D- = 0.
        criteria = pd.DataFrame({"Criterion": ["Skill", "Price"], "Type": ["benefit", "cost"], "Weight": [.5, .5], "Unit": ["points", "KZT"]})
        matrix = pd.DataFrame({"Alternative": ["A", "B"], "Skill": [3, 4], "Price": [4, 3]})
        result = evaluate(matrix, criteria, min_alternatives=2, min_criteria=2)
        np.testing.assert_allclose(result.topsis_normalized, [[.6, .8], [.8, .6]])
        np.testing.assert_allclose(result.topsis_weighted, [[.3, .4], [.4, .3]])
        np.testing.assert_allclose(result.comparison["WSM score"], [0, 1])
        np.testing.assert_allclose(result.comparison["TOPSIS score"], [0, 1])
        np.testing.assert_allclose(result.comparison["Distance to best"], [np.sqrt(.02), 0])

    def test_baseline_and_manual_example(self):
        result = evaluate(self.matrix, self.criteria)
        np.testing.assert_allclose(result.comparison["WSM score"], [.5766666666666667, .5833333333333333, .3666666666666667, .64, .35])
        self.assertEqual(result.comparison.loc[result.comparison["TOPSIS rank"].eq(1), "Alternative"].item(), "Software Engineering")
        self.criteria["Weight"] = [.10, .10, .30, .30, .20]
        result = evaluate(self.matrix, self.criteria)
        self.assertEqual(result.comparison.loc[result.comparison["WSM rank"].eq(1), "Alternative"].item(), "AI / ML")
        self.assertEqual(result.comparison.loc[result.comparison["TOPSIS rank"].eq(1), "Alternative"].item(), "Data Science")

    def test_equal_options_tie_and_zero_column(self):
        self.matrix.iloc[:, 1:] = [0, 10, 5, 5, 5]
        result = evaluate(self.matrix, self.criteria)
        np.testing.assert_allclose(result.comparison["WSM score"], 1)
        np.testing.assert_allclose(result.comparison["TOPSIS score"], .5)
        self.assertTrue(result.comparison["TOPSIS rank"].eq(1).all())
        self.assertTrue(np.isfinite(result.topsis_normalized).all().all())

    def test_bad_weights(self):
        for value in [-.1, np.nan, np.inf, "bad", .3]:
            with self.subTest(value=value):
                bad = self.criteria.astype(object)
                bad.loc[0, "Weight"] = value
                with self.assertRaises(ValidationError):
                    evaluate(self.matrix, bad)

    def test_bad_matrix_values(self):
        for value in [np.nan, np.inf, -1, "bad"]:
            with self.subTest(value=value):
                bad = self.matrix.astype(object)
                bad.loc[0, "Total cost"] = value
                with self.assertRaises(ValidationError):
                    evaluate(bad, self.criteria)

    def test_names_shape_type_and_rating_bounds(self):
        bad = self.matrix.copy()
        bad.loc[0, "Alternative"] = " cloud / devops "
        with self.assertRaises(ValidationError):
            evaluate(bad, self.criteria)
        for bad in [self.matrix.iloc[:4], self.matrix.drop(columns="Career fit")]:
            with self.assertRaises(ValidationError):
                evaluate(bad, self.criteria)
        bad = self.criteria.copy()
        bad.loc[0, "Type"] = "neither"
        with self.assertRaises(ValidationError):
            evaluate(self.matrix, bad)
        bad = self.matrix.copy()
        bad.loc[0, "Interest fit"] = 11
        with self.assertRaises(ValidationError):
            evaluate(bad, self.criteria)

    def test_units_do_not_change_rankings(self):
        original = evaluate(self.matrix, self.criteria)
        self.matrix["Total cost"] = self.matrix["Total cost"] / 1000
        rescaled = evaluate(self.matrix, self.criteria)
        np.testing.assert_allclose(original.comparison["WSM score"], rescaled.comparison["WSM score"])
        np.testing.assert_allclose(original.comparison["TOPSIS score"], rescaled.comparison["TOPSIS score"])


if __name__ == "__main__":
    unittest.main()
