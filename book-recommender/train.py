import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import warnings

#warnings.filterwarnings('ignore')
#%matplotlib inline

import tensorflow.keras as tf
from sklearn.model_selection import train_test_split

# read datasets
ratings_df = pd.read_csv("data/ratings.csv") 
books_df = pd.read_csv("data/books.csv")

print('----- RATINGS:')
print(ratings_df.head())
print(ratings_df.shape)
print(ratings_df.user_id.nunique())
print(ratings_df.book_id.nunique())
print(ratings_df.isna().sum())
print('----- BOOKS:')
print(books_df.head())

# setup training datasets
Xtrain, Xtest = train_test_split(ratings_df, test_size=0.2, random_state=1)
print(f"Shape of train data: {Xtrain.shape}")
print(f"Shape of test data: {Xtest.shape}")

# Setup the model

#Get the number of unique entities in books and users columns
nbook_id = ratings_df.book_id.nunique()
nuser_id = ratings_df.user_id.nunique()

print(f"nbook_id: {nbook_id}")
print(f"nuser_id: {nuser_id}")

DIMENSIONS = 5 #15

#Book input network
input_books = tf.layers.Input(shape=[1])
embed_books = tf.layers.Embedding(nbook_id + 1, DIMENSIONS)(input_books)
books_out = tf.layers.Flatten()(embed_books)

#user input network
input_users = tf.layers.Input(shape=[1])
embed_users = tf.layers.Embedding(nuser_id + 1, DIMENSIONS)(input_users)
users_out = tf.layers.Flatten()(embed_users)

DENSE_LAYER_SIZE = 64 # 128

conc_layer = tf.layers.Concatenate()([books_out, users_out])
x = tf.layers.Dense(DENSE_LAYER_SIZE, activation='relu')(conc_layer)
x_out = x = tf.layers.Dense(1, activation='relu')(x)
model = tf.Model([input_books, input_users], x_out)

opt = tf.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=opt, loss='mean_squared_error')
model.summary()

# train the model - SLOW

hist = model.fit([Xtrain.book_id, Xtrain.user_id], Xtrain.rating, 
                 batch_size=64, 
                 epochs=3, #5, 
                 verbose=1,
                 validation_data=([Xtest.book_id, Xtest.user_id], Xtest.rating))

#save the model
model.save('model')

# Extract embeddings for analysis for use here: http://projector.tensorflow.org/
book_em = model.get_layer('embedding')
book_em_weights = book_em.get_weights()[0]
book_em_weights.shape

books_df_copy = books_df.copy()
books_df_copy = books_df_copy.set_index("book_id")

b_id =list(ratings_df.book_id.unique())
b_id.remove(10000)
dict_map = {}
for i in b_id:
    dict_map[i] = books_df_copy.iloc[i]['title']
    
out_v = open('vecs.tsv', 'w')
out_m = open('meta.tsv', 'w')
for i in b_id:
    book = dict_map[i]
    embeddings = book_em_weights[i]
    out_m.write(book + "\n")
    out_v.write('\t'.join([str(x) for x in embeddings]) + "\n")
    
out_v.close()
out_m.close()

# view history as a chart
train_loss = hist.history['loss']
val_loss = hist.history['val_loss']
plt.plot(train_loss, color='r', label='Train Loss')
plt.plot(val_loss, color='b', label='Validation Loss')
plt.title("Train and Validation Loss Curve")
plt.legend()
plt.show()

print('done')