
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

# prepare the data
ids = []
listed_in = []
for index, row in titles_df.iterrows():
    id = int(row.show_id[1:])
    ids.append(id)
    listed_in.append(set(row.listed_in.split(', ')))

#encoded_ids = to_categorical(ids) # one hot encode ids

mlb = MultiLabelBinarizer() # one hot encoding for multiple labels
encoded_listed_in = mlb.fit(listed_in)

similar_categories = ['Sci-Fi & Fantasy'] # what categry's are we interested in?
encoded_categories = mlb.transform(similar_categories)

pred = model.predict(encoded_categories)
pred = pred.reshape(-1) #reshape to single dimension

# get 10 shows in the similar category
pred_ids = (-pred).argsort()[0:10]
print(pred_ids)

print(titles_df.iloc[pred_ids])

print('done')
