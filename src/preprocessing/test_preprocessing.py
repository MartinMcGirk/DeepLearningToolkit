from unittest import TestCase

import math

from src.preprocessing.data_preprocessor import DataPreprocessor
from src.preprocessing.preprocessor_options import PreprocessorOptions


class PreProcessingTestCase(TestCase):

    def setUp(self):
        self.file = 'test_data.csv'

    def test_preprocessor_can_read_from_file_and_split_into_sets(self):
        preprocessor = DataPreprocessor()
        preprocessor_options = PreprocessorOptions(
            file=self.file,
            numerical_columns=[],
            categorical_columns=[],
            autofill_data=False,
            encode_categories=False,
            feature_scaling=False
        )
        X_train, X_test, y_train, y_test = preprocessor.process(preprocessor_options)
        self.assertIsNotNone(X_train)
        self.assertIsNotNone(X_test)
        self.assertIsNotNone(y_train)
        self.assertIsNotNone(y_test)
        self.assertEquals(len(X_train[:, 0]), 8)
        self.assertEquals(len(X_test[:, 0]), 2)
        self.assertEquals(len(y_train), 8)
        self.assertEquals(len(y_test), 2)
