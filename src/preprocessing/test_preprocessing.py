from unittest import TestCase

import math

from src.preprocessing.data_preprocessor import DataPreprocessor
from src.preprocessing.preprocessor_options import PreprocessorOptions


class PreProcessingTestCase(TestCase):

    def setUp(self):
        self.preprocessor = DataPreprocessor()
        self.file = 'test_data.csv'
        self.preprocessor_options = PreprocessorOptions(
            self.file,
            [[1,3]],
            [0],
            True,
            True,
            True
        )

    def test_preprocessing_exists(self):
        self.preprocessor.process(self.preprocessor_options)
        self.assertIsNotNone(self.preprocessor.X_train)
        self.assertIsNotNone(self.preprocessor.X_test)
        self.assertIsNotNone(self.preprocessor.y_train)
        self.assertIsNotNone(self.preprocessor.y_test)

    def test_can_get_dataset_from_csv(self):
        self.preprocessor._get_dataset_from_csv(self.file)
        self.assertIsNotNone(self.preprocessor.X)
        self.assertEquals(self.preprocessor.X[0, 0], 'France')
        self.assertEquals(self.preprocessor.X[1, 1], 27)
        self.assertIsNotNone(self.preprocessor.y)
        self.assertEquals(self.preprocessor.y[0], 'No')
        self.assertEquals(self.preprocessor.y[1], 'Yes')

    def test_can_autofill_missing_data(self):
        self.preprocessor._get_dataset_from_csv(self.file)
        self.assertTrue(math.isnan(self.preprocessor.X[4, 2]))
        self.preprocessor._autofill_missing_data([[1, 3]])
        self.assertFalse(math.isnan(self.preprocessor.X[4, 2]))

    def test_can_encode_categorical_data(self):
        self.preprocessor._get_dataset_from_csv(self.file)
        self.preprocessor._autofill_missing_data([[1, 3]])
        self.assertEquals(self.preprocessor.X[0, 0], 'France')
        self.preprocessor._encode_categorical_data([0])
        self.assertNotEqual(self.preprocessor.X[0, 0], 'France')