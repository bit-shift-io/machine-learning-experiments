import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow import keras
from sklearn import preprocessing
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import accuracy_score

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
y = show_id_oh # expected outputs for training

# get the model
def get_model(n_inputs, n_outputs):
    DENSE_LAYER_SIZE = 20
    inp = keras.layers.Input(shape=(n_inputs,))
    de = keras.layers.Dense(DENSE_LAYER_SIZE, activation='relu')(inp) #
    dr = keras.layers.Dropout(.2)(de)
    out = keras.layers.Dense(n_outputs, activation='sigmoid')(dr)
    model = keras.models.Model(inp, out)
    opt = keras.optimizers.Adam(learning_rate=0.01)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['categorical_accuracy'])
    return model

# evaluate a model using repeated k-fold cross-validation
def evaluate_model(X, y):
	results = list()
	n_inputs, n_outputs = X.shape[1], y.shape[1]
	# define evaluation procedure
	cv = RepeatedKFold(n_splits=4, n_repeats=3, random_state=1)
	# enumerate folds
	for train_ix, test_ix in cv.split(X):
		# prepare data
		X_train, X_test = X[train_ix], X[test_ix]
		y_train, y_test = y[train_ix], y[test_ix]
		# define model
		model = get_model(n_inputs, n_outputs)
		# fit model
		model.fit(X_train, y_train, verbose=0, epochs=100)
		# make a prediction on the test set
		yhat = model.predict(X_test)
		# round probabilities to class labels
		yhat = yhat.round()
		# calculate accuracy
		acc = accuracy_score(y_test, yhat)
		# store result
		print('>%.3f' % acc)
		results.append(acc)
	return results


# evaluate model & summarize performance
results = evaluate_model(x, y)
print('Model Accuracy: %.3f (%.3f)' % (np.mean(results), np.std(results)))

model = get_model(x.shape[1], y.shape[1])
print(model.summary())

# learn the data
history = model.fit(x, y, epochs=100, verbose=1)

# run some predictions - what shows have the following categories?
categories = [
    ['Documentaries'],
    ['Drama'],
    ['Documentaries', 'Drama']
]
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
cat_acc = history.history['categorical_accuracy']
plt.plot(train_loss, color='r', label='Train Loss')
plt.plot(cat_acc, color='g', label='Train Accuracy')
plt.xlabel("Epochs")
plt.title("Train and Validation Loss Curve")
plt.legend()
plt.grid()
plt.show()


print('done')