from src.preprocessing.data_preprocessor import DataPreprocessor
from src.preprocessing.preprocessor import Preprocessor


class PreprocessorBuilder():

    def build_preprocessor(self, preprocessor_options):
        data_preprocessor = DataPreprocessor()
        return Preprocessor(preprocessor=data_preprocessor)
