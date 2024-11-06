## code inspired by tensorflow 2.0 tutorial --> freecodecamp
## changed substantially due to the 'estimator' module no longer functioning with tensorflow 2.16.2
## comments added to show understanding of core concepts

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

# Separate the 'survived' column to create the target variables
y_train = dftrain.pop('survived').astype(np.float32)  # Ensure label is float32
y_eval = dfeval.pop('survived').astype(np.float32)    # Ensure label is float32

# Ensure categorical columns are of type string
for feature_name in ['sex', 'n_siblings_spouses', 'parch', 'class', 'deck', 'embark_town', 'alone']:
    dftrain[feature_name] = dftrain[feature_name].astype(str)
    dfeval[feature_name] = dfeval[feature_name].astype(str)

## categorizes all of the categorical data in the dataset 
CATEGORICAL_COLUMNS = ['sex', 'n_siblings_spouses', 'parch', 'class', 'deck', 'embark_town', 'alone']

## categorizes all of the numerical data in the dataset
NUMERIC_COLUMNS = ['age', 'fare']

## initializes an empty dictionary that will store the feature definitions for each feature in the dataset
inputs = {}
## loops through each column in 'CATEGORICAL_COLUMNS' and holds the current column value being processed --> sex, class, etc
for feature_name in CATEGORICAL_COLUMNS:
    ## creates an input layer for categorical features, represented as strings for TensorFlow compatibility
    inputs[feature_name] = tf.keras.layers.Input(shape=(1,), name=feature_name, dtype=tf.string)

## loops through each column in 'NUMERIC_COLUMNS' and holds the current column value being processed --> age, fare
for feature_name in NUMERIC_COLUMNS:
    ## creates an input layer for numerical features, represented as float32 for TensorFlow compatibility
    inputs[feature_name] = tf.keras.layers.Input(shape=(1,), name=feature_name, dtype=tf.float32)

## initializes an empty list to hold encoded features
encoded_features = []
## loops through each column in 'CATEGORICAL_COLUMNS' to process categorical data
for feature_name in CATEGORICAL_COLUMNS:
    ## extracts the possible values associated with a column --> sex = ['male', 'female']
    vocabulary = dftrain[feature_name].unique()
    ## uses TensorFlow's StringLookup to convert categorical values to one-hot encoded vectors based on the vocabulary list --> 'Male' = 0 and 'Female' to 1 (or vice versa)
    layer = tf.keras.layers.StringLookup(vocabulary=list(vocabulary), output_mode='one_hot')
    ## applies the encoding to the input layer and appends to encoded_features
    ## adds the changed feature to the data frame inside of tensor flow
    encoded_feature = layer(inputs[feature_name])
    encoded_features.append(encoded_feature)

## loops through each column in 'NUMERIC_COLUMNS' to normalize numerical data
for feature_name in NUMERIC_COLUMNS:
    ## creates a normalization layer that standardizes numeric values to have mean=0 and variance=1
    normalizer = tf.keras.layers.Normalization(axis=None)
    ## adapts the normalization layer based on training data values in the specific numeric column
    normalizer.adapt(dftrain[feature_name].values)
    ## applies the normalizer to the input layer and appends to encoded_features
    encoded_feature = normalizer(inputs[feature_name])
    encoded_features.append(encoded_feature)

## combines all of the processed categorical and numerical features into a single tensor
all_features = tf.keras.layers.concatenate(encoded_features)

## defines the Keras model architecture, which consists of a single dense output layer with a sigmoid activation
x = tf.keras.layers.Dense(1, activation='sigmoid')(all_features)
## creates a Keras model that accepts the inputs dictionary and outputs the final prediction through the dense layer
model = tf.keras.Model(inputs=inputs, outputs=x)

## compiles the model with the Adam optimizer, binary cross-entropy loss, and accuracy as the evaluation metric
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

## creates the input function suitable for the data being fed in. 'data_df' = pandas dataframe. 'label_df' = the labels (y_train, y_eval). 'num_epochs' = number of times data should be repeated during training. 'shuffle' = boolean that determines if data is shuffled or not. 'batch_size' = number of samples per batch
def make_input_fn(data_df, label_df, shuffle=True, batch_size=32, num_epochs=10):
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
## trains the model by looking at the attributes of people on the ship (dftrain), and whether they survived or not (y_train)
train_input_fn = make_input_fn(dftrain, y_train, shuffle=True, num_epochs=10)
## evaluates the model by looking at the attributes of people on the ship (dfeval), and whether they survived or not (y_eval)
eval_input_fn = make_input_fn(dfeval, y_eval, shuffle=False, num_epochs=1)

## trains the model using the data provided by train_input_fn, for 10 epochs and with 10 steps per epoch
model.fit(train_input_fn(), epochs=10, steps_per_epoch=10)

## evaluates the model on the evaluation dataset provided by eval_input_fn, using 10 steps
result = model.evaluate(eval_input_fn(), steps=10)

clear_output()
## prints the accuracy result of the model evaluation
print(f"Accuracy: {result[1]:.4f}")
