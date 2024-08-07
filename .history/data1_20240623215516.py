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

def CreateDatasetFromEntries(entries):
      x_train = np.stack([entries[time] for time in list(entries.keys())[0:-1]])
      y_train = np.stack([entries[time][0] for time in list(entries.keys())[1:]])
      return x_train, y_train
      


if __name__ == "__main__":
      data = FetchDataYF()
      indicators = FetchIndicators()
      entries = TransformDataIntoSequence1(data, indicators)
      x_train, y_train = CreateDatasetFromEntries(entries)

      model = keras.Sequential()
      model.add(keras.Input(shape=(3, )))
      model.add(layers)
      model.add(layers.Dense(1))
      model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4),
                    loss=keras.losses.MeanSquaredError())
      model.summary()
      model.fit(x_train, y_train, batch_size=4, epochs=20, verbose=1, validation_split=0.2)

      print(x_train[-2:])
      print(model(x_train[-2:]))
      print(y_train[-2:])
