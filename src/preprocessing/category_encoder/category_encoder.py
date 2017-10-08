import numpy as np


class CategoryEncoder:

    def encode_categorical_data(self, X, y, categorical_columns):
        """Replaces columns of categorical data with multiple numerical category columns
    
        Keyword arguments:
            X -- A numpy 2 dimensional ndarray of values.
            y -- A numpy ndarray of values. Supports only 2 unique values currently.
            categorical_columns -- An array of integer column indexes.
        """

        # Encoding categorical data
        from sklearn.preprocessing import LabelEncoder, OneHotEncoder

        indexes_to_delete = []

        for cc in categorical_columns:
            labelEncoder_X = LabelEncoder()
            X[:, cc] = labelEncoder_X.fit_transform(X[:, cc])
            # Add the last generated column index to an array
            # to be deleted later to avoid the variable trap
            if indexes_to_delete == []:
                indexes_to_delete.append(max(X[:, cc]))
            else:
                indexes_to_delete.append(max(X[:, cc]) + max(indexes_to_delete) + 1)


        onehotencoder = OneHotEncoder(categorical_features=categorical_columns)
        X = onehotencoder.fit_transform(X).toarray()
        # Avoid the variable trap
        X = np.delete(X, indexes_to_delete, 1)

        labelEncoder_Y = LabelEncoder()
        y = labelEncoder_Y.fit_transform(y)
        return X, y