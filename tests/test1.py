#import os
#os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from tensorflow import keras


path = "/Users/MOPOLLIKA/Downloads/mnist.npz"
mnist = keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data(path)
x_train, x_test = x_train / 255.0, x_test / 255.0
print(tf.shape(x_train))

model = keras.models.Sequential()
model.add(keras.Input(shape=(28,28))) # shape - (sequence_length, input_size)
model.add(keras.layers.LSTM(40, return_sequences=False))
model.add(keras.layers.Dense(10))


print(model.summary())

loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optim = keras.optimizers.Adam(learning_rate=0.001)
metrics = ["accuracy"]

model.compile(loss=loss, optimizer=optim, metrics=metrics)

batch_size = 50
epochs = 5

model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=2)

model.evaluate(x_test, y_test, batch_size=batch_size, verbose=2)