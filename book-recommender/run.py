
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import warnings

from tensorflow import keras

model = keras.models.load_model('model')

# read datasets
ratings_df = pd.read_csv("data/ratings.csv") 
books_df = pd.read_csv("data/books.csv")

b_id =list(ratings_df.book_id.unique())
b_id.remove(10000)

#Making recommendations for user 100
USER = 100

book_arr = np.array(b_id) #get all book IDs
user = np.array([USER for i in range(len(b_id))])
pred = model.predict([book_arr, user])
print(pred)

pred = pred.reshape(-1) #reshape to single dimension
pred_ids = (-pred).argsort()[0:5]
print(pred_ids)

print(books_df.iloc[pred_ids])

web_book_data = books_df[["book_id", "title", "image_url", "authors"]]
web_book_data = web_book_data.sort_values('book_id')
print(web_book_data.head())

web_book_data.to_json(r'web_book_data.json', orient='records')