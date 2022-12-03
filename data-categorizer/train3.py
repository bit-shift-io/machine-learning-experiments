import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from sklearn import preprocessing
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer

# read datasets, then only grab first n rows for a smaller dataset
products_df = pd.read_json("data/products")
print(products_df.head())


# https://www.ritchieng.com/machinelearning-one-hot-encoding/
# convert id's to labels, which then get converted to onehot encoding
le = preprocessing.LabelEncoder()
id_l = le.fit_transform(products_df.product_id) #['s1', 's2' ,'s3']) #products_df.show_id)
print(id_l)
ohe = preprocessing.OneHotEncoder()
id_oh = ohe.fit_transform(id_l.reshape(-1, 1)).toarray()
print(id_oh)


# compile training data
x_1 = products_df.category #encoded_category # inputs training dataset
x_2 = products_df.ingredients # ingredients_vec
y = id_oh # expected outputs for training


# https://www.muratkarakaya.net/2022/11/part-d-preprocessing-text-with-tf-data.html?m=1
@tf.keras.utils.register_keras_serializable()
def custom_standardization(input_string):
	str = tf.strings.lower(input_string, encoding='utf-8')
	str = tf.strings.regex_replace(str, "\(.*\)", "") # remove stuff in brackets
	str = tf.strings.regex_replace(str, "[^,\w]|\d", "") # remove non word characters (except comma) and remove numbers
	str = tf.strings.regex_replace(str, ",", " ") # comma to space
	return str


# Create a vectorization layer and adapt it to the text
category_tv = keras.layers.TextVectorization(
    standardize=custom_standardization,
    output_mode="binary", #tf-idf / int / binary / count
)
category_tv.adapt(products_df.category)
vocab = category_tv.get_vocabulary()  
print(vocab)
print("2 sample text preprocessing:")
print(" Given raw data: " )
print(tf.expand_dims(products_df.category[0], -1))
tokenized = category_tv(tf.expand_dims(products_df.category[0], -1))
print(" Tokenized and Transformed to a vector of integers: ", tokenized )


ingredients_tv = keras.layers.TextVectorization(
    standardize=custom_standardization,
    output_mode="binary", #tf-idf / int / binary / count
)
ingredients_tv.adapt(products_df.ingredients)
ingredients_vocab = ingredients_tv.get_vocabulary()  
print(ingredients_vocab)
print(len(ingredients_vocab))

# get the model
def get_model(category_tv, ingredients_tv, n_outputs):
	DENSE_LAYER_SIZE = 20

	inp1 = tf.keras.Input(shape=(1,), dtype=tf.string)
	vec1 = category_tv(inp1)

	de1 = keras.layers.Dense(DENSE_LAYER_SIZE, activation='relu')(vec1) #inp1) #
	dr1 = keras.layers.Dropout(.1)(de1)

	inp2 = tf.keras.Input(shape=(1,), dtype=tf.string)
	vec2 = ingredients_tv(inp2)
	de2 = keras.layers.Dense(DENSE_LAYER_SIZE, activation='relu')(vec2) #
	dr2 = keras.layers.Dropout(.1)(de2)

	rs2 = keras.layers.Rescaling(0.01)(dr2) # reduce impact of input 2
	conc = keras.layers.Concatenate()([dr1, rs2])

	out = keras.layers.Dense(n_outputs, activation='sigmoid')(conc) # softmax or sigmoid - doesnt seem to make a diff
	model = keras.models.Model([inp1, inp2], out)
	opt = keras.optimizers.Adam(learning_rate=0.01)
	model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['categorical_accuracy'])
	return model

model = get_model(category_tv, ingredients_tv, y.shape[1])
print(model.summary())


# learn the data
history = model.fit([x_1, x_2], y, epochs=500, verbose=1)


# run some predictions - lets feed in a show we know about and see what other shows are like it...
row_idx = 0
row = products_df.iloc[row_idx]

cat = tf.expand_dims(row.category, -1) 
ing = tf.expand_dims(row.ingredients, -1) 
pred = model.predict([cat, ing])
pred = np.array(pred).reshape(-1) #reshape to single dimension

pred_ids = (-pred).argsort()[0:10]
print(f'\nTop similar products to {row.product_id} are... ')
print('categories: ', row.category)
print('ingredients: ', row.ingredients)
for pred_idx in pred_ids:
	show_id = le.inverse_transform([pred_idx])[0]
	print(f"{show_id} has {pred[pred_idx]} chance being similar to {row.product_id}")

print('\n')


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