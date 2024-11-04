from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import clear_output
from six.moves import urllib
import tensorflow as tf
from pathlib import Path

fc = tf.feature_column

train_path = Path("/your/path/to/train.csv")
eval_path = Path("/your/path/to/eval.csv")

dftrain = pd.read_csv(train_path)
dfeval = pd.read_csv(eval_path)

y_train = dftrain.pop('survived').astype(np.float32)
y_eval = dfeval.pop('survived').astype(np.float32)

for feature_name in ['sex', 'n_siblings_spouses', 'parch', 'class', 'deck', 'embark_town', 'alone']:
    dftrain[feature_name] = dftrain[feature_name].astype(str)
    dfeval[feature_name] = dfeval[feature_name].astype(str)

CATEGORICAL_COLUMNS = ['sex', 'n_siblings_spouses', 'parch', 'class', 'deck', 'embark_town', 'alone']
NUMERIC_COLUMNS = ['age', 'fare']

inputs = {}
for feature_name in CATEGORICAL_COLUMNS:
    inputs[feature_name] = tf.keras.layers.Input(shape=(1,), name=feature_name, dtype=tf.string)

for feature_name in NUMERIC_COLUMNS:
    inputs[feature_name] = tf.keras.layers.Input(shape=(1,), name=feature_name, dtype=tf.float32)

encoded_features = []
for feature_name in CATEGORICAL_COLUMNS:
    vocabulary = dftrain[feature_name].unique()
    layer = tf.keras.layers.StringLookup(vocabulary=list(vocabulary), output_mode='one_hot')
    encoded_feature = layer(inputs[feature_name])
    encoded_features.append(encoded_feature)

for feature_name in NUMERIC_COLUMNS:
    normalizer = tf.keras.layers.Normalization(axis=None)
    normalizer.adapt(dftrain[feature_name].values)
    encoded_feature = normalizer(inputs[feature_name])
    encoded_features.append(encoded_feature)

all_features = tf.keras.layers.concatenate(encoded_features)

x = tf.keras.layers.Dense(1, activation='sigmoid')(all_features)
model = tf.keras.Model(inputs=inputs, outputs=x)

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

def make_input_fn(data_df, label_df, shuffle=True, batch_size=32, num_epochs=10):
    def input_function():
        ds = tf.data.Dataset.from_tensor_slices((dict(data_df), label_df))
        if shuffle:
            ds = ds.shuffle(1000)
        ds = ds.batch(batch_size).repeat(num_epochs)
        return ds
    return input_function

train_input_fn = make_input_fn(dftrain, y_train, shuffle=True, num_epochs=10)
eval_input_fn = make_input_fn(dfeval, y_eval, shuffle=False, num_epochs=1)

model.fit(train_input_fn(), epochs=10, steps_per_epoch=10)

result = model.evaluate(eval_input_fn(), steps=10)

print(f"Accuracy: {result[1]:.4f}")
