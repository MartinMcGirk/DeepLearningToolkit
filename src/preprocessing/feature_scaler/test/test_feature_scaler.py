from unittest import TestCase

import numpy as np
import pandas as pd

from src.preprocessing.feature_scaler.feature_scaler import FeatureScaler


class FeatureScalerTestCase(TestCase):
    def setUp(self):
        data = np.array([['', 'Country', 'Age', 'Salary', 'Bonus', 'y'],
                         ['Row1', 0, 44, 72000, 1000, 'No'],
                         ['Row2', 1, 36, 48000, 1200, 'Yes'],
                         ['Row3', 0, 30, 54000, 500, 'No'],
                         ['Row4', 0, 38, 61000, 2000, 'No'],
                         ['Row5', 1, 40, 32000, 4500, 'Yes'],
                         ['Row6', 0, 44, 72000, 600, 'Yes']
                         ])
        dataset = pd.DataFrame(
            data=data[1:, 1:],
            index=data[1:, 0],
            columns=data[0, 1:]
        )
        self.X = dataset.iloc[:, :-1].values

    def test_dataset_gets_feature_scaled(self):
        scaler = FeatureScaler()
        scaled_data = scaler.apply_feature_scaling(self.X)

        for row in scaled_data:
            self.assertTrue(max(row) < 3)
            self.assertTrue(min(row) > -3)