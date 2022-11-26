import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow import keras
from sklearn import preprocessing
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer

# read datasets, then only grab first n rows for a smaller dataset
#titles_df = pd.read_csv("data/netflix_titles.csv") 
titles_df = pd.read_csv("test-data/test-data-1.csv")
titles_df = titles_df.iloc[:100]
print(titles_df.head())

# https://stackoverflow.com/questions/57073029/remove-stopwords-from-most-common-words-from-set-of-sentences-in-python
desc_cv = CountVectorizer(min_df=1, lowercase=True, stop_words='english')
desc_vec = desc_cv.fit_transform(titles_df.description).toarray()


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
x_1 = encoded_listed_in # inputs training dataset
x_2 = desc_vec
y = show_id_oh # expected outputs for training

# get the model
def get_model(n_inputs_1, n_inputs_2, n_outputs):
	DENSE_LAYER_SIZE = 20
	inp1 = keras.layers.Input(shape=(n_inputs_1,))
	de1 = keras.layers.Dense(DENSE_LAYER_SIZE, activation='relu')(inp1) #
	dr1 = keras.layers.Dropout(.2)(de1)

	inp2 = keras.layers.Input(shape=(n_inputs_2,))
	de2 = keras.layers.Dense(DENSE_LAYER_SIZE, activation='relu')(inp2) #
	dr2 = keras.layers.Dropout(.2)(de2)

	conc = keras.layers.Concatenate()([dr1, dr2])

	out = keras.layers.Dense(n_outputs, activation='sigmoid')(conc)
	model = keras.models.Model([inp1, inp2], out)
	opt = keras.optimizers.Adam(learning_rate=0.01)
	model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['categorical_accuracy'])
	return model

model = get_model(x_1.shape[1], x_2.shape[1], y.shape[1])
print(model.summary())

# learn the data
history = model.fit([x_1, x_2], y, epochs=100, verbose=1)


# run some predictions - lets feed in a show we know about and see what other shows are like it...
show_idx = 3
s_id_oh = y[show_idx]
s_id = le.inverse_transform(ohe.inverse_transform([s_id_oh]))[0]
x_test_1 = np.array(x_1[show_idx]).reshape(-1, x_1[show_idx].shape[0])
x_test_2 = np.array(x_2[show_idx]).reshape(-1, x_2[show_idx].shape[0])

pred = model.predict([x_test_1, x_test_2])
pred = np.array(pred).reshape(-1) #reshape to single dimension

print(titles_df.head())

pred_ids = (-pred).argsort()[0:10]
print(f'\nTop similar shows to {s_id} are {pred_ids}\n')
for pred_idx in pred_ids:
	show_id = le.inverse_transform([pred_idx])[0]
	print(f"{show_id} has {pred[pred_idx]} chance being similar to {s_id}")

print('\n')

#for pred_idx, pred_chance in enumerate(pred):
#	show_id = le.inverse_transform([pred_idx])[0]
#	print(f"{show_id} has {pred_chance} chance of having categories: {categories[cat_idx]}")

#print(pred)

# save the model
model.save('model')

# report
train_loss = history.history['loss']
cat_acc = history.history['categorical_accuracy']
plt.plot(train_loss, color='r', label='Train Loss')
plt.plot(cat_acc, color='g', label='Train Accuracy')
plt.xlabel("Epochs")
plt.title("Train and Validation Loss Curve")
plt.legend()
plt.grid()
plt.show()


print('done')