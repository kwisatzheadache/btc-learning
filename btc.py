import numpy as np
import tensorflow as tf
from sklearn.feature_extraction import DictVectorizer 
import json
import scipy
from sklearn.preprocessing import normalize

vec = DictVectorizer()

with open('./btc-history.json') as history:
    data = json.load(history)
    print(data)

historyVectorized = vec.fit_transform(data).toarray()

print(historyVectorized.shape)

normal = normalize(historyVectorized)

print(normal.shape)
print(scipy.stats.describe(normal))
