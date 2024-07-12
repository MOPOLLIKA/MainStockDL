import keras
import numpy as np
from keras import layers
from data import FetchDataYF, FetchIndicators
from sklearn.preprocessing import StandardScaler

def CommonKeyValues(dct1, dct2):
      keys1 = set(dct1.keys())
      keys2 = set(dct2.keys())
      commonKeys = sorted(list(keys1 & keys2))
      dct1 = {time: dct1[time] for time in commonKeys}
      dct2 = {time: dct2[time] for time in commonKeys}
      return dct1, dct2

def TransformDataIntoSequence1(data, indicators):
      """
      Simple transformation of data into a sequence of [close, volume, indicator] values attached to time keys in a dictionary.
      """
      data, indicators = CommonKeyValues(data, indicators)
      closes = {time: data[time][3] for time in data}
      volumes = {time: data[time][5] for time in data}
      indicators = {time: indicators[time][0] for time in data} # take only obv values
      entries = {time: [closes[time], volumes[time], indicators[time]] for time in data}
      return entries

def TransformEntries(entries, timestep):
      # Stopped here. The next step is to implement timestep feature and test different indicators to see how it will improve the accuracy and record everything
      # to some table or just keep a diary of what was changing
      
      for time in entries:

      pass

def CreateDatasetFromEntries(entries):
      x_train = np.stack(np.array([np.reshape(entries[time], (1, -1)) for time in list(entries.keys())[0:-1]]))
      print(x_train[0:10])
      y_train = np.stack([entries[time][0] for time in list(entries.keys())[1:]])
      return x_train, y_train
      


if __name__ == "__main__":
      data = FetchDataYF()
      indicators = FetchIndicators()
      entries = TransformDataIntoSequence1(data, indicators)
      x_train, y_train = CreateDatasetFromEntries(entries)

      model = keras.Sequential()
      model.add(keras.Input(shape=(1, 3, )))
      model.add(layers.Dense(128))
      model.add(layers.Conv1D(32, 3))
      model.add(layers.Dense(1))
      model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4),
                    loss=keras.losses.MeanSquaredError())
      model.summary()
      earlyStopping = keras.callbacks.EarlyStopping("val_loss", patience=5)
      model.fit(x_train, y_train, batch_size=4, epochs=20, callbacks=earlyStopping, verbose=1, validation_split=0.2)

      print(x_train[-2:])
      print(model(x_train[-2:]))
      print(y_train[-2:])
