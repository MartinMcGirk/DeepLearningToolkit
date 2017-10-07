
class CategoryEncoder:

    def encode_categorical_data(self, X, y, categorical_columns):
        """Replaces columns of categorical data with multiple numerical category columns
    
        Keyword arguments:
        categorical_columns -- An array of integer column indexes.
        """

        # Encoding categorical data
        from sklearn.preprocessing import LabelEncoder, OneHotEncoder

        for cc in categorical_columns:
            labelEncoder_X = LabelEncoder()
            X[:, cc] = labelEncoder_X.fit_transform(X[:, cc])
        onehotencoder = OneHotEncoder(categorical_features=categorical_columns)
        X = onehotencoder.fit_transform(X).toarray()
        X = X[:, 1:]
        labelEncoder_Y = LabelEncoder()
        y = labelEncoder_Y.fit_transform(y)
        return X, y