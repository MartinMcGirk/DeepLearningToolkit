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
        X_train, X_test, y_train, y_test = preprocessor.set_up_preprocessor_from_base_data(preprocessor_options)
        self.assertIsNotNone(X_train)
        self.assertIsNotNone(X_test)
        self.assertIsNotNone(y_train)
        self.assertIsNotNone(y_test)
        self.assertEquals(len(X_train[:, 0]), 8)
        self.assertEquals(len(X_test[:, 0]), 2)
        self.assertEquals(len(y_train), 8)
        self.assertEquals(len(y_test), 2)

    def test_data_is_not_autofilled_if_not_asked_for(self):
        preprocessor = DataPreprocessor()
        preprocessor_options = PreprocessorOptions(
            file='test_data_with_missing_values.csv',
            numerical_columns=[],
            categorical_columns=[],
            autofill_data=False,
            encode_categories=False,
            feature_scaling=False
        )
        X_train, X_test, y_train, y_test = preprocessor.set_up_preprocessor_from_base_data(preprocessor_options)
        contains_nan = [math.isnan(float(val)) for val in X_train[:, 1]]
        self.assertTrue(True in contains_nan)

    def test_data_is_autofilled_if_asked_for(self):
        preprocessor = DataPreprocessor()
        preprocessor_options = PreprocessorOptions(
            file='test_data_with_missing_values.csv',
            numerical_columns=[1, 2],
            categorical_columns=[],
            autofill_data=True,
            encode_categories=False,
            feature_scaling=False
        )
        X_train, X_test, y_train, y_test = preprocessor.set_up_preprocessor_from_base_data(preprocessor_options)
        col_1_contains_nan = [math.isnan(float(val)) for val in X_train[:, 1]]
        self.assertTrue(True not in col_1_contains_nan)
        col_2_contains_nan = [math.isnan(float(val)) for val in X_train[:, 2]]
        self.assertTrue(True not in col_2_contains_nan)

    def test_data_is_not_encoded_if_not_asked_for(self):
        preprocessor = DataPreprocessor()
        preprocessor_options = PreprocessorOptions(
            file=self.file,
            numerical_columns=[],
            categorical_columns=[],
            autofill_data=False,
            encode_categories=False,
            feature_scaling=False
        )
        X_train, X_test, y_train, y_test = preprocessor.set_up_preprocessor_from_base_data(preprocessor_options)
        contains_string = [isinstance(val, str) for val in X_train[:, 0]]
        self.assertTrue(False not in contains_string)

    def test_data_is_encoded_if_asked_for(self):
        preprocessor = DataPreprocessor()
        preprocessor_options = PreprocessorOptions(
            file=self.file,
            numerical_columns=[1, 2],
            categorical_columns=[0],
            autofill_data=False,
            encode_categories=True,
            feature_scaling=False
        )
        X_train, X_test, y_train, y_test = preprocessor.set_up_preprocessor_from_base_data(preprocessor_options)
        contains_string = [isinstance(val, str) for val in X_train[:, 0]]
        self.assertTrue(True not in contains_string)

    def test_can_use_alternate_category_encoder_if_needed(self):
        class SpyEncoder:
            def __init__(self):
                self.called = False

            def encode_categorical_data(self, X, y, categorical_columns):
                self.called = True
                return X, y

        encoder = SpyEncoder()
        preprocessor = DataPreprocessor(category_encoder=encoder)
        preprocessor_options = PreprocessorOptions(
            file=self.file,
            numerical_columns=[1, 2],
            categorical_columns=[0],
            autofill_data=False,
            encode_categories=True,
            feature_scaling=False
        )
        preprocessor.set_up_preprocessor_from_base_data(preprocessor_options)
        self.assertTrue(encoder.called)

    def test_can_use_alternate_data_autofiller_if_needed(self):
        class SpyFiller:
            def __init__(self):
                self.called = False

            def autofill_data(self, X, numerical_columns):
                self.called = True
                return X

        autofiller = SpyFiller()
        preprocessor = DataPreprocessor(data_auto_filler=autofiller)
        preprocessor_options = PreprocessorOptions(
            file=self.file,
            numerical_columns=[1, 2],
            categorical_columns=[0],
            autofill_data=True,
            encode_categories=False,
            feature_scaling=False
        )
        preprocessor.set_up_preprocessor_from_base_data(preprocessor_options)
        self.assertTrue(autofiller.called)

    def test_can_use_alternate_feature_scaler_if_needed(self):
        class SpyScaler:
            def __init__(self):
                self.called = False

            def apply_feature_scaling(self, X):
                self.called = True
                return X

        scaler = SpyScaler()
        preprocessor = DataPreprocessor(feature_scaler=scaler)
        preprocessor_options = PreprocessorOptions(
            file=self.file,
            numerical_columns=[1, 2],
            categorical_columns=[0],
            autofill_data=False,
            encode_categories=False,
            feature_scaling=True
        )
        preprocessor.set_up_preprocessor_from_base_data(preprocessor_options)
        self.assertTrue(scaler.called)

