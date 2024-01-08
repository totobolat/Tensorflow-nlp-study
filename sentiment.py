import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
import keras
from keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import layers
from tensorflow.keras import losses
import re
import string


df = pd.read_csv('/kaggle/input/copy-of-nlp-getting-started/train.csv')
pd.set_option('display.max_columns',None)
df.tail()

sns.countplot(x = df['target'], color= 'green')
pd.isna(df).sum()
df.describe()

df_train = df.sample(frac=0.8, random_state=0)
df_val = df.drop(df_train.index)

def custom_standardization(input_data):
    lowercase = tf.strings.lower(input_data)
    stripped_html = tf.strings.regex_replace(lowercase, '<br />', ' ')
    return tf.strings.regex_replace(stripped_html,
                                  '[%s]' % re.escape(string.punctuation),
                                  '')

max_features = 10000
sequence_length = 250

vectorize_layer = layers.TextVectorization(
    standardize=custom_standardization,
    max_tokens=max_features,
    output_mode='int',
    output_sequence_length=sequence_length)

dataset1 = tf.data.Dataset.from_tensor_slices((df_val['text'], df_val['target']))
dataset1 = dataset1.map(lambda x, y: {'text': x, 'target': y})

def convert_ds_to_tuple(sample):
    return sample['text'], sample['target']

df_validate = dataset1.map(convert_ds_to_tuple).batch(32)



dataset = tf.data.Dataset.from_tensor_slices((df_train['text'], df_train['target']))
dataset = dataset.map(lambda x, y: {'text': x, 'target': y})

def convert_ds_to_tuple(sample):
    return sample['text'], sample['target']

dataset = dataset.map(convert_ds_to_tuple).batch(32)

train_text = dataset.map(lambda x,y: x)
vectorize_layer.adapt(train_text)
df_validate1 = df_validate.map(lambda x,y: x)
vectorize_layer.adapt(df_validate1)

def vectorize_text(text, label):
    text = tf.expand_dims(text, -1)
    return vectorize_layer(text), label

# retrieve a batch (of 32 reviews and labels) from the dataset
text_batch, label_batch = next(iter(dataset))
first_review, first_label = text_batch[0], label_batch[0]
print("Review", first_review)
#print("Label", dataset[first_label])
print("Vectorized review", vectorize_text(first_review, first_label))

train_ds = dataset.map(vectorize_text)
val_ds = df_validate.map(vectorize_text)

embedding_dim = 16
model = tf.keras.Sequential([
  layers.Embedding(max_features, embedding_dim),
  layers.Dropout(0.2),
  layers.GlobalAveragePooling1D(),
  layers.Dropout(0.2),
  layers.Dense(1)])

model.summary()

model.compile(loss=losses.BinaryCrossentropy(),
              optimizer='adam',
              metrics=tf.metrics.BinaryAccuracy(threshold=0.0))

epochs = 100
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs)

loss, accuracy = model.evaluate(val_ds)

print("Loss: ", loss)
print("Accuracy: ", accuracy)
export_model = tf.keras.Sequential([
  vectorize_layer,
  model,
  layers.Activation('sigmoid')
])

export_model.compile(
    loss=losses.BinaryCrossentropy(from_logits=False), optimizer="adam", metrics=['accuracy']
)

