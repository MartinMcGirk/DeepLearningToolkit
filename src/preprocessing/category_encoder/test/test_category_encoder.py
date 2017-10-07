from unittest import TestCase
import pandas as pd
import numpy as np

from src.preprocessing.category_encoder.category_encoder import CategoryEncoder


class CategoryEncoderTestCase(TestCase):
    def test_category_cols_with_two_values_split_into_one_col(self):
        data = np.array([['', 'CatCol', 'y'],
                         ['Row1', 'One', 'Yes'],
                         ['Row2', 'Two', 'No'],
                         ['Row3', 'One', 'No'],
                         ['Row4', 'Two', 'Yes']
                         ])

        X, y = self._call_encoder_with(data)

        self.assertEquals(len(X[0]), 1)

    def test_category_cols_with_three_values_split_into_two_cols(self):
        data = np.array([['','CatCol','y'],
                         ['Row1', 'One', 'Yes'],
                         ['Row2', 'Two', 'No'],
                         ['Row3', 'Two', 'No'],
                         ['Row4', 'Three', 'Yes']
                         ])

        X, y = self._call_encoder_with(data)

        self.assertEquals(len(X[0]), 2)

    def test_category_cols_with_five_values_split_into_four_cols(self):
        data = np.array([['','CatCol','y'],
                         ['Row1', 'One', 'Yes'],
                         ['Row2', 'Two', 'No'],
                         ['Row3', 'Three', 'No'],
                         ['Row4', 'Four', 'Yes'],
                         ['Row4', 'Five', 'Yes']
                         ])

        X, y = self._call_encoder_with(data)

        self.assertEquals(len(X[0]), 4)

    def test_category_col_gets_encoded_into_numbers(self):
        data = np.array([['', 'CatCol', 'y'],
                         ['Row1', 'One', 'Yes'],
                         ['Row2', 'Two', 'No'],
                         ['Row3', 'One', 'No'],
                         ['Row4', 'Two', 'Yes']
                         ])
        output = [
            [0],
            [1],
            [0],
            [1]
        ]

        X, y = self._call_encoder_with(data)

        self.assertTrue(self._data_is_same(X, output))

    def test_category_col_with_three_values_gets_encoded_into_numbers(self):
        data = np.array([['', 'CatCol', 'y'],
                         ['Row1', 'One', 'Yes'],
                         ['Row2', 'Two', 'No'],
                         ['Row3', 'Three', 'No'],
                         ['Row4', 'Two', 'Yes']
                         ])
        output = [
            [0, 0],
            [0, 1],
            [1, 0],
            [0, 1]
        ]

        X, y = self._call_encoder_with(data)

        self.assertTrue(self._data_is_same(X, output))

    def _call_encoder_with(self, data):
        input_X, input_y = self._get_dataset_from_nparray(data)
        encoder = CategoryEncoder()
        X, y = encoder.encode_categorical_data(
            input_X,
            input_y,
            [0]
        )

        return X, y

    def _get_dataset_from_nparray(self, data):
        dataset = pd.DataFrame(
            data=data[1:, 1:],
            index=data[1:, 0],
            columns=data[0, 1:]
        )
        X = dataset.iloc[:, :-1].values
        y = dataset.iloc[:, -1].values
        return X, y

    def _data_is_same(self, a, b):
        return np.array_equal(a, b)