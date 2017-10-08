from unittest import TestCase
import pandas as pd
import numpy as np

from src.preprocessing.data_autofiller.data_autofiller import DataAutofiller


class DataAutofillerTestCase(TestCase):
    def setUp(self):
        data = np.array([['', 'Country', 'Age', 'Salary', 'Bonus', 'y'],
                         ['Row1', 'France', 44, 72000, 1000, 'No'],
                         ['Row2', 'Spain', np.NaN, 48000, 1200, 'Yes'],
                         ['Row3', 'Germany', 30, 54000, 500, 'No'],
                         ['Row4', 'Spain', 38, 61000, 2000, 'No'],
                         ['Row5', 'Germany', 40, np.NaN, 4500, 'Yes'],
                         ['Row6', 'France', 44, 72000, 600, 'Yes']
                         ])
        dataset = pd.DataFrame(
            data=data[1:, 1:],
            index=data[1:, 0],
            columns=data[0, 1:]
        )
        self.X = dataset.iloc[:, :-1].values

    def test_one_column_with_missing_data_gets_autofilled_but_other_does_not(self):
        autofiller = DataAutofiller()
        X = autofiller.autofill_data(self.X, [1])
        self.assertNotEquals(X[1, 1], 'nan')
        self.assertEquals(X[4, 2], 'nan')

    def test_that_multiple_columns_with_missing_data_can_be_autofilled(self):
        autofiller = DataAutofiller()
        X = autofiller.autofill_data(self.X, [1, 2])
        self.assertNotEquals(X[4, 2], 'nan')

    def test_that_columns_with_no_missing_data_are_not_modified(self):
        autofiller = DataAutofiller()
        X = autofiller.autofill_data(self.X, [3])
        self.assertEquals(X[0, 3], 1000)
        self.assertEquals(X[1, 3], 1200)
        self.assertEquals(X[2, 3], 500)
        self.assertEquals(X[3, 3], 2000)
        self.assertEquals(X[4, 3], 4500)
        self.assertEquals(X[5, 3], 600)
