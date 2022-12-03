
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import warnings

from tensorflow import keras
from keras.utils import to_categorical
from sklearn.preprocessing import MultiLabelBinarizer

model = keras.models.load_model('model')

# read datasets
titles_df = pd.read_csv("data/netflix_titles.csv") 

# Only grab first n rows for a smaller dataset
titles_df = titles_df.iloc[:100]

print(titles_df.head())

# prepare the data
onehot_show_ids = pd.get_dummies(titles_df.show_id)

# create multi-label one hot encoding of all listed_in/categories
listed_in = []
for index, row in titles_df.iterrows():
    listed_in.append(set(row.listed_in.split(', ')))

mlb = MultiLabelBinarizer() # one hot encoding for multiple labels
encoded_listed_in = mlb.fit_transform(listed_in)
listed_in_width = encoded_listed_in.shape[1]
print(mlb.classes_)

similar_categories = [['Documentaries']] # what categry's are we interested in finding similar shows?
encoded_categories = mlb.transform(similar_categories)
print(encoded_categories)
pred = model.predict(encoded_categories)
#print(pred)

# invert one hot encoding back to show id mapped against the chance this show overlaps your categories of interest
results = {}
for idx, x in enumerate(pred):
    show_id = titles_df.show_id[idx]
    results[show_id] = x[0][0]
    
sorted_results = sorted(results.items(), key=lambda item: -item[1])

# just print out the top 10
#sorted_results = sorted_results.iloc[:10]
for idx, x in enumerate(sorted_results):
    row = titles_df.loc[titles_df['show_id'] == x[0]]
    reduced_row = row[['show_id', 'listed_in']] # 'title',
    print(reduced_row, 'PROBABILITY: ', x[1])
    

print('done')
