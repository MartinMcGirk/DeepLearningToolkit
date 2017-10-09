import pandas as pd

from src.preprocessing.category_encoder.category_encoder import CategoryEncoder
from src.preprocessing.data_autofiller.data_autofiller import DataAutofiller
from src.preprocessing.feature_scaler.feature_scaler import FeatureScaler


class DataPreprocessor():
    def __init__(self, category_encoder=None, data_auto_filler=None, feature_scaler=None):
        self.category_encoder = category_encoder or CategoryEncoder()
        self.data_auto_filler = data_auto_filler or DataAutofiller()
        self.feature_scaler = feature_scaler or FeatureScaler()

    def process(self, preprocessing_options):
        X, y = self._get_dataset_from_csv(preprocessing_options.file)

        if preprocessing_options.autofill_data:
            X = self._autofill_missing_data(X, preprocessing_options.numerical_columns)

        if preprocessing_options.encode_categories:
            X, y = self._encode_categorical_data(X, y, preprocessing_options.categorical_columns)

        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

        if preprocessing_options.feature_scaling:
            X_train, X_test = self._apply_feature_scaling(X_train, X_test)

        return X_train, X_test, y_train, y_test

    def _get_dataset_from_csv(self, file):
        dataset = pd.read_csv(file)
        X = dataset.iloc[:, :-1].values
        y = dataset.iloc[:, -1].values
        return X, y

    def _autofill_missing_data(self, X, numerical_columns):
        return self.data_auto_filler.autofill_data(X, numerical_columns)

    def _encode_categorical_data(self, X, y, categorical_columns):
        X, y = self.category_encoder.encode_categorical_data(
            X=X,
            y=y,
            categorical_columns=categorical_columns
        )
        return X, y

    def _apply_feature_scaling(self, X_train, X_test):
        X_train = self.feature_scaler.apply_feature_scaling(X_train)
        X_test = self.feature_scaler.apply_feature_scaling(X_test)
        return X_train, X_test
