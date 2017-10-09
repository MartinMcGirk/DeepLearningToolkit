# DeepLearningToolkit
A toolkit to abstract away the complexity of getting started quickly with deep learning

### Currently Available Functionality
#### Data Preprocessing

##### Capabilities

The preprocessing module allows for the hassle free application of data preprocessing tasks that must be carried out ahead of feeding the data into a deep learning model.
 
Included in the module is the ability to:
 - work with data given to it in the form of CSV
 - autofill missing data points in columns of numerical data (to the mean value of the column)
 - encode categorical text data into split out columns of numerical data, automatically dropping a column each time to avoid the variable trap
 - feature scaling to shrink all the values of the dataset into a similar scale that the deep learning model can use to then process efficiently
 
Requirements about the data to be preprocessed:
 - Data should supplied in CSV format
 - Text categorical columns cannot have missing values
 
##### Usage:

Data preprocessing can be instigated by calling the `process` function on the `DataPreprocessor` class and describing your data to it using a `PreprocessorOptions` object

```python
    preprocessor = DataPreprocessor()
    preprocessor_options = PreprocessorOptions(
        file='fileName.csv', # Path to your dataset as a CSV
        numerical_columns=[0, 3, 5], # Indexes of columns of numerical data in your dataset
        categorical_columns=[1, 2], # Indexes of columns of Text categorical data in your dataset
        autofill_data=True, # Should autofilling of missing data be applied
        encode_categories=True, # Should your categorical data be encoded and split out
        feature_scaling=True # Should feature scaling be applied
    )
    X_train, X_test, y_train, y_test = preprocessor.process(preprocessor_options)
```

### Planned Functionality
 - Image preprocessing
 - ANN spin up
