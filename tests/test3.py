import tensorflow as tf
from tensorflow import keras
# model.compile - set hyper parameters and use various techniques
# model.fit - train the model 
# model.evaluate - check the testing data
# model.predict - make a prediction

# subclass model neural network implementation

class Model1(keras.Model):
  def __init__(self):
    super().__init__()
    self.inputs = keras.Input(shape=(28, 28))
    self.dense1 = keras.layers.Dense(20)
    self.dense2 = keras.layers.Dense(20)
    self.dropout = keras.layers.Dropout(0.5)
  
  def call(self, inputs, training=False):
    inputs = self.inputs(inputs)
    x = self.dense1(inputs)
    x = self.dropout(x, training=training)
    return self.dense2(x)
  
model1 = Model1()

# functional API implementation

inputs = keras.Input(shape=(28, 28))
x = keras.layers.Dense(50, activation='relu')(inputs)
outputs = keras.layers.Dense(10, activation='softmax')(x)
model2 = keras.Model(inputs=inputs, outputs=outputs)

# another API implementation of CNN RGB image classificator

inputs = keras.Input(shape=(28, 28))
#processed = keras.layers.RandomCrop(width=28, height=28)(inputs)
conv = keras.layers.Conv1D(filters=30, kernel_size=2)(inputs)
pooling = keras.layers.GlobalAveragePooling1D()(conv)
feature = keras.layers.Dense(10)(pooling)

model3 = keras.Model(inputs, feature)
#backbone = keras.Model(processed, conv)
activations = keras.Model(conv, feature)

# sequential implementation

model4 = keras.Sequential([
  keras.Input(shape=(None, None, 3)),
  keras.layers.Conv2D(filters=32, kernel_size=3)
])

# parameter functions

loss = keras.losses.SparseCategoricalCrossentropy(from_logits=False)
optim = keras.optimizers.Adam(learning_rate=0.0002)
metrics = ["accuracy"]

# mnist dataset

path = "/Users/MOPOLLIKA/Downloads/mnist.npz"
mnist = keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data(path)
x_train, x_test = x_train / 255.0, x_test / 255.0

# initialize the neural network -- which for some reason doesn't work

#model1.compile(loss=loss, optimizer=optim, metrics=metrics)

#model1.fit(x_train, y_train, batch_size=30, epochs=10, verbose=2)

#model1.evaluate(x_test, y_test, batch_size=30, verbose=2)

# initialize the second one -- fail

#model2.summary()

#model2.compile(loss=loss, optimizer=optim, metrics=metrics)

#model2.fit(x_train, y_train, batch_size=30, epochs=10, verbose=2)

#model2.evaluate(x_test, y_test, batch_size=30, verbose=2)

# initialize the third one -- WORKS! at 0.2 accuracy

#model3.compile(loss=loss, optimizer=optim, metrics=metrics)

#model3.fit(x_train, y_train, batch_size=10, epochs=15, verbose=2)

#model3.evaluate(x_test, y_test, batch_size=30, verbose=2)

lst = [1, 5, 2, 7]
print(map(
  lambda x: 2*x,
  lst
))