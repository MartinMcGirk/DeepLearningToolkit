from sklearn.preprocessing import Imputer


class DataAutofiller:
    def autofill_data(self, data, numerical_columns):
        """Replaces missing values in columns of numerical data with the mean value of the column.

        Keyword arguments:
            data: A 2 dimensional ndarray of values
            numerical_columns -- An array of arrays. Each inner array should contain the indexes of numerical columns
        """
        for nc in numerical_columns:
            imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
            imputer = imputer.fit(data[:, nc:nc + 1])
            data[:, nc:nc + 1] = imputer.transform(data[:, nc:nc + 1])
        return data