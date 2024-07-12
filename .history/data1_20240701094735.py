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
      indicators = {time: indicators[time][7] for time in data} # take only obv values
      entries = {time: [closes[time], volumes[time], indicators[time]] for time in data}
      array = list(map(lambda x: list(x), list(np.reshape(list(closes.values()), (-1, 1)))))
      print(StandardizeArraySK(array, 50))
      return entries

def TransformEntries(entries, timestep, stride):
      values = list(entries.values())
      times = list(entries.keys())
      entries = {}
      for index in range(0, len(values) - timestep + 1, stride):
            entries[times[index]] = values[index:index + timestep]
      return entries

def StandardizeArraySK(array, nValues):
      length = len(array)
      numberOfChunks = length // nValues
      arrayNew = []

      if numberOfChunks == 0:
            scaler = StandardScaler().fit(array)
            standardized = scaler.transform(array)
            arrayNew = standardized
            return arrayNew
      
      for index in range(numberOfChunks):
            chunk = array[index * nValues:(index + 1) * nValues]
            scaler = StandardScaler().fit(chunk)
            standardized = scaler.transform(chunk)
            arrayNew += standardized

      number = length / nValues
      if int(number) < number:
            chunkLast = array[length - nValues:]
            scaler = StandardScaler().fit(chunkLast)
            standardized = scaler.transform(chunkLast)
            arrayNew += list(standardized)
      return arrayNew


def CreateDatasetFromEntries(entries):
      x_train = np.stack(np.array([entries[time] for time in list(entries.keys())[0:-1]]))
      y_train = []
      for time in list(entries.keys())[1:]:
            entry = np.reshape(entries[time], (-1, 3))
            y_train.append([[array[0] for array in entry]])
      y_train = np.stack(y_train)
      return x_train, y_train
      
def TestModel(model: keras.Model, x_train, y_train):
      print(x_train[-2:])
      print(model(x_train[-2:]))
      print(y_train[-2:])
      return True

if __name__ == "__main__":
      data = FetchDataYF()
      indicators = FetchIndicators()
      entries = TransformDataIntoSequence1(data, indicators)
      entries = TransformEntries(entries, 3, 2)
      x_train, y_train = CreateDatasetFromEntries(entries)

      model = keras.Sequential()
      model.add(keras.Input(shape=(None, 3, )))
      model.add(layers.Dense(64))
      model.add(layers.Dense(1))

      model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-5), loss=keras.losses.MeanSquaredError())
      model.summary()
      earlyStopping = keras.callbacks.EarlyStopping("val_loss", patience=5, restore_best_weights=True)
      model.fit(x_train, y_train, batch_size=4, epochs=50, callbacks=earlyStopping, verbose=1, validation_split=0.2)

      TestModel(model, x_train, y_train)

      
