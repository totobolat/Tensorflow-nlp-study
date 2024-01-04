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