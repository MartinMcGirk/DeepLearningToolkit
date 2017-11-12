from src.neural_network.artificial_neural_network_builder import ArtificialNeuralNetworkBuilder
from src.neural_network.artificial_neural_network_options import ArtificialNeuralNetworkOptions
from src.preprocessing.preprocessor_builder import PreprocessorBuilder
from src.preprocessing.preprocessor_options import PreprocessorOptions

#
# This file doesn't work at the moment, this meant more as
# a template to show how the finished article will work
#

preprocessing_options = PreprocessorOptions(
    file='preprocessing/test_data_multi_category.csv',
    numerical_columns=[[2,4]],
    categorical_columns=[0,1],
    autofill_data=True,
    encode_categories=True,
    feature_scaling=True
)
ann_options = ArtificialNeuralNetworkOptions(
    input_dimension=12
)
# Preprocess data
data_preprocessor_builder = PreprocessorBuilder()
data_preprocessor = data_preprocessor_builder.build_preprocessor()
X_train, X_test, y_train, y_test = data_preprocessor.process(preprocessing_options)
# Create  and train artificial neural net
ann_builder = ArtificialNeuralNetworkBuilder()
ann = ann_builder.build_artificial_neural_network(ann_options)
ann.train_neural_network(X_train=X_train, y_train=y_train)
# TODO: predict results based on trained model