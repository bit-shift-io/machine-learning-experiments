import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow import keras
from sklearn import preprocessing

# read datasets, then only grab first n rows for a smaller dataset
#titles_df = pd.read_csv("data/netflix_titles.csv") 
titles_df = pd.read_csv("test-data/test-data-1.csv")
titles_df = titles_df.iloc[:100]
print(titles_df.head())

# https://www.ritchieng.com/machinelearning-one-hot-encoding/
# convert id's to labels, which then get converted to onehot encoding
le = preprocessing.LabelEncoder()
show_id_l = le.fit_transform(titles_df.show_id) #['s1', 's2' ,'s3']) #titles_df.show_id)
print(show_id_l)
ohe = preprocessing.OneHotEncoder()
show_id_oh = ohe.fit_transform(show_id_l.reshape(-1, 1)).toarray()
print(show_id_oh)

# convert the column 'listed_in' which is a multi-label column (i.e. tags) to use
# one hot encoding
listed_in = []
for index, row in titles_df.iterrows():
    listed_in.append(list(set(row.listed_in.split(', '))))

mlb = preprocessing.MultiLabelBinarizer() # one hot encoding for multiple labels
encoded_listed_in = mlb.fit_transform(listed_in)
listed_in_width = encoded_listed_in.shape[1]
print(mlb.classes_)

# compile training data
x = encoded_listed_in # inputs training dataset
y_list = show_id_oh # expected outputs for training
sample_weight = {} # dictionary of weights for each output layer for training set

for index, row in titles_df.iterrows():
    row_show_id_le = le.transform([row.show_id])
    row_show_id_ohe = ohe.transform(row_show_id_le.reshape(-1, 1)).toarray()
    print(row_show_id_le)
    sample_weight[row.show_id] = row_show_id_ohe.reshape(-1, 1)
    

# construct the model
inp = keras.layers.Input(shape=(listed_in_width,))
out_layers = []

for index, row in titles_df.iterrows():
    out = keras.layers.Dense(1, name=row.show_id)(inp)
    out_layers.append(out)

model = keras.models.Model(inp, out_layers)
model.compile(loss='mse', optimizer='adam')
print(model.summary())

# learn the data
history = model.fit(x, y_list, epochs=500, verbose=1, sample_weight=sample_weight)

# run a prediction - what shows have the following categories?
categories = [
    ['Documentaries'],
    ['Drama'],
    ['Documentaries', 'Drama']
 ] #['Documentaries'] #['Drama'] #['Documentaries', 'Drama']
categories_enc = mlb.transform(categories)
print(categories_enc)
print(titles_df.head())
for cat_idx, cat_enc in enumerate(categories_enc):
    reshaped = cat_enc.reshape(-1, cat_enc.shape[0])
    pred = model.predict(reshaped)

    pred = np.array(pred).reshape(-1) #reshape to single dimension
    #pred_ids = (-pred).argsort() #[0:5]
    #print(pred_ids)

    for pred_idx, pred_chance in enumerate(pred):
        show_id = le.inverse_transform([pred_idx])[0]
        print(f"{show_id} has {pred_chance} chance of having categories: {categories[cat_idx]}")

    #print(pred)

# save the model
model.save('model')

# report
train_loss = history.history['loss']
plt.plot(train_loss, color='r', label='Train Loss')
plt.xlabel("Epochs")
plt.title("Train and Validation Loss Curve")
plt.legend()
plt.grid()
plt.show()


print('done')