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

# Only grab first n rows for a smaller dataset
titles_df = titles_df.iloc[:100]

print(titles_df.head())

# prepare the data
ids = []
listed_in = []
out_layers = []

onehot_show_ids = pd.get_dummies(titles_df.show_id)
#print(onehot_show_ids)
#onehot_show_ids = MultiLabelBinarizer().fit_transform(shows_ids.values)


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
#train_x, test_x, train_y, test_y = train_test_split(encoded_listed_in, encoded_ids, test_size=0.2, random_state=1)

# https://stackoverflow.com/questions/57349349/using-different-sample-weights-for-each-output-in-a-multi-output-keras-model

inp = keras.layers.Input(shape=(listed_in_width,))

out_layers = []

x = encoded_listed_in # inputs
y_list = [] # expected outputs
sample_weight = {} # dictionary of weights for each output layer



for index, row in titles_df.iterrows():
    onehot_id = onehot_show_ids[row.show_id]
    sample_weight[row.show_id] = onehot_id
    y_list.append(onehot_id)
    
for index, row in titles_df.iterrows():
    out = keras.layers.Dense(1, name=row.show_id)(inp)
    out_layers.append(out)

model = keras.models.Model(inp, out_layers)
model.compile(loss='mse',
              optimizer='adam')


# TODO: add validation_data - training data!
history = model.fit(x, y_list, epochs=5, batch_size=32, verbose=1, sample_weight=sample_weight)


#save the model
model.save('model')

'''
inp = keras.layers.Input(shape=(5,))
# assign names to output layers for more clarity
out1 = keras.layers.Dense(1, name='out1')(inp)
out2 = keras.layers.Dense(1, name='out2')(inp)

model = keras.models.Model(inp, [out1, out2])
model.compile(loss='mse',
              optimizer='adam')



# create some dummy training data as well as sample weight
n_samples = 100
X = np.random.rand(n_samples, 5)
y1 = np.random.rand(n_samples,1)
y2 = np.random.rand(n_samples,1)

w1 = np.random.rand(n_samples,)
w2 = np.random.rand(n_samples,)

model.fit(X, [y1, y2], epochs=5, batch_size=16, sample_weight={'out1': w1, 'out2': w2})




# assign names to output layers for more clarity
#out1 = layers.Dense(1, name='out1')(inp)
#out2 = layers.Dense(1, name='out2')(inp)

model = Sequential()
model.add(Dense(100, activation='relu', input_dim=listed_in_width)) #train_x.shape[1]))
model.add(Dropout(0.1))
model.add(Dense(50, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(encoded_ids.shape[1], activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer="adam", metrics=["categorical_accuracy"])

history = model.fit(encoded_listed_in, encoded_ids, sample_weight=encoded_ids, epochs=5, batch_size=2000)

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

'''
print('done')