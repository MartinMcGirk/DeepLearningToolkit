from unittest import TestCase

from src.preprocessing.data_preprocessor import DataPreprocessor
from src.preprocessing.preprocessor import Preprocessor
from src.preprocessing.preprocessor_builder import PreprocessorBuilder
from src.preprocessing.preprocessor_options import PreprocessorOptions


class data_preprocessor_builder(TestCase):

    def test_can_initialise_preprocessor_builder(self):
        builder = PreprocessorBuilder()
        self.assertIsNotNone(builder)

    def test_can_return_preprocessor_when_given_options(self):
        builder = PreprocessorBuilder()
        options = PreprocessorOptions(
            file='test_data.csv',
            numerical_columns=[],
            categorical_columns=[],
            autofill_data=False,
            encode_categories=False,
            feature_scaling=False
        )
        preprocessor = builder.build_preprocessor(options)
        self.assertIsNotNone(preprocessor)
        self.assertIsInstance(preprocessor, Preprocessor)

    def test_returns_preprocessor_with_embedded_data_preprocessor(self):
        builder = PreprocessorBuilder()
        options = PreprocessorOptions(
            file='test_data.csv',
            numerical_columns=[],
            categorical_columns=[],
            autofill_data=False,
            encode_categories=False,
            feature_scaling=False
        )
        preprocessor = builder.build_preprocessor(options)
        self.assertIsInstance(preprocessor.preprocessor, DataPreprocessor)