import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import warnings
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.optimizers import SGD

from keras.utils import to_categorical

# read datasets
titles_df = pd.read_csv("data/netflix_titles.csv") 
print(titles_df.head())

# prepare the data
ids = []
listed_in = []
for index, row in titles_df.iterrows():
    id = int(row.show_id[1:])
    ids.append(id)
    listed_in.append(set(row.listed_in.split(', ')))

encoded_ids = to_categorical(ids) #titles_df.shape[0]) # one hot encode ids

mlb = MultiLabelBinarizer() # one hot encoding for multiple labels
encoded_listed_in = mlb.fit_transform(listed_in)
listed_in_width = encoded_listed_in.shape[1]
print(mlb.classes_)

#d = {'show_id': ids, 'listed_in': encoded_listed_in}
#prepared_df = pd.DataFrame(data=d)

#dataset = tf.data.Dataset.from_tensor_slices(
#        (ids, encoded_listed_in)
#    )
#dataset = encoded_listed_in

#train_x, test_x = train_test_split(dataset, test_size=0.2, random_state=1)

# _x is the inputs, _y is the outputs
train_x, test_x, train_y, test_y = train_test_split(encoded_listed_in, encoded_ids, test_size=0.2, random_state=1)

model = Sequential()
model.add(Dense(5000, activation='relu', input_dim=train_x.shape[1]))
model.add(Dropout(0.1))
model.add(Dense(600, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(train_y.shape[1], activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer="adam", metrics=["categorical_accuracy"])

history = model.fit(train_x, train_y, epochs=5, batch_size=2000)

model.save('model')

train_loss = history.history['loss']
#val_loss = history.history['val_loss']
cat_acc = history.history['categorical_accuracy']
plt.plot(train_loss, color='r', label='Train Loss')
#plt.plot(val_loss, color='b', label='Validation Loss')
plt.plot(cat_acc, color='g', label='Categorical Accuracy')
plt.xlabel("Epochs")
plt.title("Train and Validation Loss Curve")
plt.legend()
plt.grid()
plt.show()


print('done')