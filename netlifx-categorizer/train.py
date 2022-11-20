import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import warnings
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer

# read datasets
titles_df = pd.read_csv("data/netflix_titles.csv") 
print(titles_df.head())



y = []
for index, row in titles_df.iterrows():
    y.append(set(row.listed_in.split(',')))

mlb = MultiLabelBinarizer()
encoded_y = mlb.fit_transform(y)
y_width = encoded_y.shape[1]

input_layer = keras.layers.Input(shape=[1])
embed_layer = keras.layers.Embedding(y_width)(input_layer)
out_layer = keras.layers.Flatten()(embed_layer)

print('done')