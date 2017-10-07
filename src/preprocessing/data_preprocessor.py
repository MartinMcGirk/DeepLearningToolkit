import pandas as pd

from src.preprocessing.category_encoder.category_encoder import CategoryEncoder


class DataPreprocessor():
    def process(self, preprocessing_options):
        self._get_dataset_from_csv(preprocessing_options.file)

        if preprocessing_options.autofill_data:
            self._autofill_missing_data(preprocessing_options.numerical_columns)

        if preprocessing_options.encode_categories:
            self._encode_categorical_data(preprocessing_options.categorical_columns)

        from sklearn.model_selection import train_test_split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=0)

        if preprocessing_options.feature_scaling:
            self._apply_feature_scaling()

    def _get_dataset_from_csv(self, file):
        dataset = pd.read_csv(file)
        self.X = dataset.iloc[:, :-1].values
        self.y = dataset.iloc[:, -1].values

    def _autofill_missing_data(self, numerical_columns):
        """Replaces missing values in columns of numerical data
        with the mean value of the column.

        Keyword arguments:
        numerical_columns -- An array of arrays. Each inner array should contain the indexes of numerical columns
        """
        from sklearn.preprocessing import Imputer
        for nc in numerical_columns:
            imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
            imputer = imputer.fit(self.X[:, nc[0]:nc[1]])
            self.X[:, nc[0]:nc[1]] = imputer.transform(self.X[:, nc[0]:nc[1]])

    def _encode_categorical_data(self, categorical_columns):
        """Replaces columns of categorical data with multiple numerical category columns

        Keyword arguments:
        categorical_columns -- An array of integer column indexes.
        """
        category_encoder = CategoryEncoder()
        self.X, self.y = category_encoder.encode_categorical_data(
            X=self.X,
            y=self.y,
            categorical_columns=categorical_columns
        )

    def _apply_feature_scaling(self):
        # Feature Scaling
        from sklearn.preprocessing import StandardScaler
        sc_X = StandardScaler()
        self.X_train = sc_X.fit_transform(self.X_train)
        self.X_test = sc_X.transform(self.X_test)
