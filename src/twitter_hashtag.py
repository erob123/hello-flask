import numpy as np
import pandas as pd
import pickle
from statistics import mode
import nltk
from nltk import word_tokenize
from tensorflow.keras.models import Model
from tensorflow.keras import models
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import plot_model
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense, Concatenate, Attention
from sklearn.model_selection import train_test_split
nltk.download('wordnet')
nltk.download('punkt')

# read the dataset file
train = pd.read_csv("train.csv")

# tweet column is input
inp_data = train["tweet"]
# target data is sentiment(s1,s2,s3,s4,s5) ,
# when (w1,w2,w3,w4) and kind(k1,k2,k3...k15)
tar_data = train.iloc[:, 4:].values

# get the column name of target
tar_lab = train.iloc[:, 4:].columns.tolist()

# value of the target label like
# s1="I can't tell" , s2="Negative" and so on till s5
# w1="current weather", w2=future forecast and so on till w4
# k1="clouds", k2="cold", k3="dry" and so on till k15
tar_lab_val = [
    "I can't tell", "Negative", "Neutral", "Positive", "Tweet not related to weather condition",
    "current (same day) weather", "future (forecast)", "I can't tell", "past weather",
    "clouds", "cold", "dry", "hot", "humid", "hurricane", "I can't tell", "ice", "other", "rain",
    "snow", "storms", "sun", "tornado", "wind"]

# clean the tweets


def clean(tweet):
    # replace and lower case the tweets
    tweet = tweet.replace(":", "").lower()
    # get only words that contains alphabets
    words = list(filter(lambda w: (w.isalpha()), tweet.split(" ")))
    # expand the shortened words
    words = [contractions[w] if w in contractions else w for w in words]
    # return all the words
    return words
