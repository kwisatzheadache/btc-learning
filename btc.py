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

float_data = normal

def generator(data, lookback, delay, min_index, max_index, shuffle=False, batch_size=128, step=6):
    if max_index is None:
        max_index = len(data) - delay - 1
    i = min_index + lookback
    while 1:
        if shuffle:
            rows = np.random.randint(min_index + lookback, max_index, size=batch_size)
        else:
            if i + batch_size >= max_index:
                i = min_index + lookback
            rows = np.arange(i, min(i + batch_size, max_index))
            i += len(rows)

        samples = np.zeros((len(rows), lookback // step, data.shape[-1]))
        targets = np.zeros((len(rows),))
        for j, row in enumerate(rows):
            indices = range(rows[j] - lookback, rows[j], step)
            samples[j] = data[indices]
            targets[j] = data[rows[j] + delay][1]
        yield samples, targets


lookback = 200
step = 2
delay = 24
batch_size = 128

train_gen = generator(float_data, lookback=lookback, delay=delay, min_index=0, max_index=1300, shuffle=True, step=step, batch_size=batch_size)
val_gen = generator(float_data, lookback=lookback, delay=delay, min_index=1301, max_index=2600, step=step, batch_size=batch_size)
test_gen = generator(float_data, lookback=lookback, delay=delay, min_index=2601, max_index=None, step=step, batch_size=batch_size)

val_steps = (2600 - 1301 - lookback) // batch_size
test_steps = (len(float_data) - 2601 - lookback) // batch_size 


from keras.models import Sequential
from keras import layers
from keras.optimizers import RMSprop

model = Sequential()
model.add(layers.Flatten(input_shape=(lookback // step, float_data.shape[-1])))
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(1))

model.compile(optimizer=RMSprop(), loss='mae')
history = model.fit_generator(train_gen, steps_per_epoch=500, epochs=20, validation_data=val_gen, validation_steps=val_steps)


execfile('./btc.py')
import matplotlib.pyplot as polt
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(loss))
polt.figure
polt.plot(epochs, loss, 'b+')
polt.plot(epochs, val_loss, 'bo')
polt.title('training and validation loss')
polt.show()
