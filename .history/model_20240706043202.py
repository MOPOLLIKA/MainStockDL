import numpy as np
import tensorflow as tf
import keras
from keras import layers
from data import CreateDataset, PredictEntries
import matplotlib as plt
import keras_tuner as kt
from keras_tuner.tuners import RandomSearch


"""
Plans:
      TASK
      1. Design a model - COMPLETED
      2. Fine-tune it 
      3. Function predicting n values forward in format: {"future_date_1": value_1, "future_date_2": value_2, ... , "future_date_n": value_n} COMPLETED
      4. Reduce all OHLC values only to C and standardize them - COMPLETED
      5. Try overnight with following parameters: units - 512, timestep - 20, lr - 0.000001
      
      TASK
      The data is very noisy which reduces the effectiveness of the model. The possible alterations to be made:
      1. Reduce the data. Leave only necessary indicators and try to combine them, so that only 3 values are left - price, volume and indicator feature, with input 
      and output masks for all of them, also the output values s COMPLETED
      2. Timestep + stride strategy seems good COMPLETED
      3. Use principal component analysis to help the prediction 
      4. Try using default scaler from scypy.preprocessing.StandardScaler, or rethink the standardization process COMPLETED

      TASK
      Another approach - try convnetting the whole sequence of candles 
      Method: Take slices of candles of some constant length as an input and output next close value
      f(X) -> Y; f( X=(batch_size, sliceLength, 3) ) -> Y=([close])
      1. Parse data into sequence of shape=(3) vectors COMPLETED
      2. Split into slices with (sliceLength) length COMPLETED
      3. Attach the slice values to x_train and close values to y_train COMPLETED

      That was the current method, here is another one:
      1. Pass the whole sequence to the model NEXT TARGET
      2. ConvNet it.

      TASK
      Create a dataset in a form of JSON file of the data for all tickers from nasdaq and also try other stock exchanges.
      The structure would be {"ticker": {"data": ..., "metadata": {"interval": "1d", "timeframe": ...}}}

      TASK
      Make fetch function update values downloaded when a day passes or maybe several days

      TASK
      Add day of the week as a feature in a format [isMonday, isTuesday, isWednesday, isThursday, isFriday] or just [isMonday, isFriday] with Boolean 0 or 1 values

      TASK
      Adjust the prediction algorithms to work with stride 1 and go one by one predicting a value and then using it for the next prediction.

      CURRENT OBJECTIVES
      1. Collect the accuracy data for all the NASDAQ tickers using 3 values and 1 value entries with the basic model.
      In order to do that a file with all the entries for daily data should be created.
      2. Abandon the 3 value strategy and complete the 1 value prediction method, and plotting method to create required graphs 
"""

### Dataset
includeIndicators = False
includeOHLC = True
includeVolume = True
#inputSize = ((np.sum(inputMask) - 1) * includeOHLC) + 1 + 10 * includeIndicators
inputSize = 12
#outputSize = np.sum(outputMask)
outputSize = 1
timestep = 6
stride = 3
((x_train, y_train), (x_test, y_test)), (priceGrowths, (lastDateIndex, commonTime)) = CreateDataset(
      ticker="NFLX",
      timeframe="DAILY",
      interval="1d",
      testFr=0.1,
      nValues=500,
      timestep=timestep,
      stride=stride,
      includeIndicators=includeIndicators,
      includeOHLC=includeOHLC,
      includeVolume=includeVolume
)
print(y_train[-1])
### Model
class StockModel(keras.Model):
      def __init__(self, inputSize, outputSize, units, unitsReduction, dropoutFr):
            # initializing parameters of the model
            super().__init__()
            self.inputSize = inputSize
            self.outputSize = outputSize
            self.units = units
            self.unitsReduction = unitsReduction
            self.dropoutFr = dropoutFr

            # Layer 1
            nUnits1 = self.units
            self.lstm1 = layers.LSTM(nUnits1, return_sequences=True)
            self.dropout1 = layers.Dropout(self.dropoutFr)
            # Layer 2
            nUnits2 = round(self.units * unitsReduction)
            self.lstm2 = layers.LSTM(nUnits2, return_sequences=True)
            self.dropout2 = layers.Dropout(self.dropoutFr)
            # Layer 3
            nUnits3 = round(self.units *  (unitsReduction ** 2))
            self.lstm3 = layers.LSTM(nUnits3, return_sequences=True)
            self.dropout3 = layers.Dropout(self.dropoutFr)
            self.denseLast = layers.Dense(outputSize)

      def call(self, inputs):
            x = self.lstm1(inputs)
            x = self.dropout1(x)

            x = self.lstm2(x)
            x = self.dropout2(x)

            x = self.lstm3(x)
            x = self.dropout3(x)

            return self.denseLast(x)
      

class StockHyperModel(kt.HyperModel):
      def build(self, hp: kt.HyperParameters):
            units = hp.Int("units", min_value=48, max_value=144, step=16)
            learning_rate = hp.Float("learning_rate", min_value=0.0001, max_value=1, step=10, sampling="log")
            model = StockModel(inputSize, outputSize, timestep, units=units, unitsReduction=0.8, dropoutFr=0.3)
            loss = keras.losses.MeanAbsolutePercentageError()
            optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
            metrics = [keras.metrics.huber]
            model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
            return model


def MakeSequentialStockModel(inputShape, outputSize, nLayers, units, dropoutRate):
      inputs = keras.Input(shape=inputShape)

      x = layers.LSTM(units, return_sequences=True)(inputs)
      x = layers.Dropout(dropoutRate)(x)
      for i in range(nLayers - 1):
            x = layers.LSTM(units, return_sequences=True)(x)
            x = layers.Dropout(dropoutRate)(x)

      outputs = layers.Dense(outputSize)(x)
      return keras.Model(inputs=inputs, outputs=outputs)


#model = StockModel(inputSize, outputSize, timestep, units=144, unitsReduction=0.8, dropoutFr=0.3)
#model = MyModel(10, 'relu')
model = MakeSequentialStockModel((timestep, inputSize), outputSize, nLayers=2, units=64, dropoutRate=0.2)
model.summary()

loss = keras.losses.MeanAbsolutePercentageError()
optimizer = keras.optimizers.Adam(learning_rate=1e-5)
metrics = [keras.metrics.MeanSquaredError]
model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

earlyStopping = keras.callbacks.EarlyStopping("val_loss", patience=10)
checkpointsPath = "/Users/MOPOLLIKA/python_StockDL/checkpoints/checkpoints2.weights.h5"
checkpoints = keras.callbacks.ModelCheckpoint(filepath=checkpointsPath, verbose=0, save_weights_only=True, save_best_only=True)
model.fit(x_train, y_train, batch_size=16, epochs=10, verbose=2, callbacks=[earlyStopping, checkpoints], validation_split=0.2)

model.evaluate(x_train, y_train, batch_size=1)




print(np.stack(x_train[len(x_train) - 3:len(x_train)]))
print(model(np.stack(x_train[len(x_train) - 3:len(x_train)])))
print(np.stack(y_train[len(y_train) - 3:len(y_train)]))
"""
### Searching for the best hyperparameters

print()
lastEntry = y_train[-1]
print(PredictEntries(model, 10, commonTime, lastDateIndex, lastEntry, "DAILY", timestep, stride))
tuner = RandomSearch(
      StockHyperModel(),
      objective="val_loss",
      max_trials=50,
      executions_per_trial=1,
      directory="stock_dir",
      project_name="stock_project7"
)

tuner.search(x_train, y_train, epochs=5, validation_split=0.2)

modelBest: keras.Model = tuner.get_best_models()[0]
hpsBest: kt.HyperParameters = tuner.get_best_hyperparameters()[0]
print(hpsBest.get("units"))
print(hpsBest.get("learning_rate"))

modelBest.evaluate(x_train, y_train, batch_size=10)

### units - 144, lr - 0.01; unitsReduction - 0.8, dropoutFr - 0.3

"""
