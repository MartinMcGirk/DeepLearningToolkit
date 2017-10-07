class PreprocessorOptions():
    def __init__(self, file, numerical_columns, categorical_columns,
                 autofill_data=True, encode_categories=True,
                 feature_scaling=True):
        self.file = file
        self.numerical_columns = numerical_columns
        self.categorical_columns = categorical_columns
        self.autofill_data = autofill_data
        self.encode_categories = encode_categories
        self.feature_scaling = feature_scaling
