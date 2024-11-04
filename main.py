from __future__ import absolute_import, division, print_function, unicode_literals

# Import necessary libraries
import numpy as np  # for multidimensional calculations
import pandas as pd  # for data manipulation
import matplotlib.pyplot as plt  # for visualization
from IPython.display import clear_output
from six.moves import urllib
import tensorflow as tf
from pathlib import Path

# Use TensorFlow feature columns
fc = tf.feature_column

# Define the file paths using Pathlib
train_path = Path("/your/path/to/train.csv")
eval_path = Path("/your/path/to/eval.csv")

# Load the datasets
dftrain = pd.read_csv(train_path)
dfeval = pd.read_csv(eval_path)

# Display the first few rows of the training data
print(dftrain.head())

# Separate the 'survived' column to create the target variables
y_train = dftrain.pop('survived')
y_eval = dfeval.pop('survived')

# Display the modified training data without the 'survived' column
print(dftrain.head())

# Show the values for the first passenger and whether they survived
print(dftrain.loc[0], y_train.loc[0])

## categorizes all of the categorical data in the dataset 
CATEGORICAL_COLUMNS = ['sex', 'n_siblings_spouses', 'parch', 'class', 'deck', 'embark_town', 'alone']

## categorizes all of the numerical data in the dataset
NUMERIC_COLUMNS = ['age', 'fare']

## initializes an empty list that will store the feature definitions for each feature in the dataset
feature_columns = []
## loops through each column in 'CATEGORICAL_COLUMNS' and holds the current column value being processed --> sex, class, etc
for feature_name in CATEGORICAL_COLUMNS:
    ## extracts the possible values associated with a column --> sex = ['male', 'female']
    vocabulary = dftrain[feature_name].unique()
    ## creates TensorFlow 'feature column object' specifically for categorical data. 'feature_name' is the name of the column and 'vocabulary' defines the unique values in that column
    ## TensorFlow matches a 'vocabulary' value to an integer --> Male = 1 -- Female = 0
    ## then the feature is appended to the feature_column --> 'feature_columns.append'
    feature_columns.append(tf.feature_column.categorical_column_with_vocabulary_list(feature_name, vocabulary))

## loops through each column in 'NUMERICAL_COLUMNS' and hold the current column value being processed --> age, fare
for feature_name in NUMERIC_COLUMNS:
    ## creates a 'numeric_column' in TensorFlow, takes the name of the feature (age, fare) and converts it to a certain type of integer. In this case 32 bit.
    feature_columns.append(tf.feature_column.numeric_column(feature_name, dtype = tf.float32))

## creates the input function suitable for the data being fed in. 'data_df' = pandas dataframe. 'label_df' = the labels (y_train, y_eval). 'num_epochs' = number of times data should be repeated during training. 'shuffle' = boolean that determines if data is shuffled or not. 'batch_size' = number of samples per batch
def make_input_fn(data_df, label_df, num_epochs=10, shuffle=True, batch_size=32):
    ## contains the main logic for the input function
    def input_function(): 
        ## converts the input data into a TensorFlow 'dataset'. '((dict(data_df), label_df))' passes the pandas dataframe along with the labels being used into the new 'tf.data.Dataset' TensorFlow dataset. 
        ds = tf.data.Dataset.from_tensor_slices((dict(data_df), label_df))
        ## shuffles the data around in order to randomize it. 
        if shuffle:
            ds = ds.shuffle(1000)
        ## splits the dataset into batches of 32 and repeats for the programmed number of epochs
        ds = ds.batch(batch_size).repeat(num_epochs) 
        return ds  # return a batch of the dataset
    return input_function  # return a function object for use

## both train and eval are being called here
## trains the model by looking at the attributes of people on the ship (dftrain), and whether they survived of not (y_train)
train_input_fn = make_input_fn(dftrain, y_train)  
## evaluates the model by looking at the attributes of people on the ship (dfeval), and whether they survived of not (y_eval)
eval_input_fn = make_input_fn(dfeval, y_eval, num_epochs=1, shuffle=False)

## the feature columns that was created above is called and placed into the 'Linear Clsssifier' module of the esimator library from TensorFlow
linear_est = tf.estimator.LinearClassifier(feature_columns=feature_columns)

## 'train_input_fn' takes 'def input_function', passes all of the information through it, and trains the model
linear_est.train(train_input_fn)
## 'result' is the output of the evaluate function, which uses the famr
result = linear_est.evaluate(eval_input_fn)

clear_output()
print(result['Accuracy'])

