from unittest import TestCase

from src.preprocessing.data_preprocessor import DataPreprocessor
from src.preprocessing.preprocessor import Preprocessor


class PreprocessorTestCase(TestCase):

    def test_can_create_preprocessor(self):
        data_preprocessor = DataPreprocessor()
        preprocessor = Preprocessor(data_preprocessor)
        self.assertIsNotNone(preprocessor)