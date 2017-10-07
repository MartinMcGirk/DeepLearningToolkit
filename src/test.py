from src.preprocessing.data_preprocessor import DataPreprocessor
from src.preprocessing.preprocessor_options import PreprocessorOptions

data_preproccessor = DataPreprocessor()
preprocessing_options = PreprocessorOptions(
    file='preprocessing/test_data_multi_category.csv',
    numerical_columns=[[2,4]],
    categorical_columns=[0,1],
    autofill_data=True,
    encode_categories=True,
    feature_scaling=True
)
data_preproccessor.process(preprocessing_options)