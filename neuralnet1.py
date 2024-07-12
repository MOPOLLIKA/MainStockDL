import numpy as np
import tensorflow as tf
import keras
from keras import layers
from data import CreateDataset
import matplotlib.pyplot as plt
import keras_tuner as kt

# open, high, low, close, volume
inputMask = [True, True, True, True, True, True]
outputMask = [False, False, False, True, False, False]
includeIndicators = True
includeOHLC = True
inputSize = np.sum(inputMask) * includeOHLC + 10 * includeIndicators
outputSize = np.sum(outputMask)
timestep = 10
stride = 1
(x_train, y_train), (x_test, y_test) = CreateDataset(
  ticker="AAPL",
  timeframe="DAILY",
  interval="1d",
  testFr=0.1,
  nValues=50,
  timestep=timestep,
  stride=stride,
  inputMask=inputMask,
  outputMask=outputMask,
  includeIndicators=includeIndicators,
  includeOHLC=includeOHLC,
)

#import sys; sys.exit()
# Model
inputs = keras.Input(shape=(timestep, inputSize))
x = layers.LSTM(128, return_sequences=True)(inputs)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(outputSize)(x)
model = keras.Model(inputs=inputs, outputs=outputs)
# Compiling

loss = keras.losses.MeanAbsolutePercentageError()
optimizer = keras.optimizers.Adam(learning_rate=0.001)
metrics = [
  #keras.metrics.MeanAbsolutePercentageError(),
]
model.summary()

model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

batch_size = 5
earlyStopping = keras.callbacks.EarlyStopping("val_loss", patience = 10, restore_best_weights=True)
history = model.fit(x_train, y_train, batch_size, 100, verbose=2, validation_split=0.1, callbacks=earlyStopping)
"""
plt.plot(history.history["loss"])
plt.title("Loss - Epoch graph")
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.show()
"""
model.evaluate(x_test, y_test, batch_size=1, verbose=2)


print(np.stack(x_train[len(x_train) - 1:len(x_train)]))
print(model(np.stack(x_train[len(x_train) - 1:len(x_train)])))
#plot the tests
"""
for lr in np.linspace(0.005, 0.0005, num=10):
      model = keras.Model(inputs=inputs, outputs=outputs)
      model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr), loss=loss, metrics=metrics)
      history = model.fit(x_train, y_train, batch_size, 10, verbose=2, validation_split=0.1)
      plt.plot(history.history["loss"])
plt.title("Loss - Epoch graph")
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.legend(list(np.linspace(0.005, 0.0005, num=10)), loc="upper right")
plt.show()
"""