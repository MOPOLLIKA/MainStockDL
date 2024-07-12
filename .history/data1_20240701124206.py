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

def DatesToValues(dates, array):
      result = {}
      count = 0
      for date in dates:
            result[date] = array[count]
            count += 1
      return result

def TransformDataIntoSequence1(data, indicators):
      """
      Simple transformation of data into a sequence of [close, volume, indicator] values attached to time keys in a dictionary.
      """
      data, indicators = CommonKeyValues(data, indicators)
      dates = list(data.keys())
      nValues = 500

      closes = {time: data[time][3] for time in data}

      volumesValues = {time: data[time][5] for time in data}
      volumesListed = list(map(lambda x: list(x), list(np.reshape(list(volumesValues.values()), (-1, 1)))))
      volumesStandardized = StandardizeArraySK(volumesListed, nValues)
      volumes = DatesToValues(dates, volumesStandardized)

      indicatorsValues = {time: indicators[time][7] for time in data} # take only obv values
      indicatorsListed = list(map(lambda x: list(x), list(np.reshape(list(indicatorsValues.values()), (-1, 1)))))
      indicatorsStandardized = StandardizeArraySK(indicatorsListed, nValues)
      indicators = DatesToValues(dates, indicatorsStandardized)

      entries = {time: [closes[time], volumes[time], indicators[time]] for time in data}
      if not len(volumesListed) == len(volumesStandardized):
            print(f"len(array): {len(volumes)}, len(standarsized): {len(volumesStandardized)}")
            raise ValueError("PIZDEC")
      for i in range(len(volumesStandardized)):
            print(f"Value_{i}: {volumesListed[i]}, standardized_{i}: {volumesStandardized[i]}")
      return entries

def TransformEntries(entries, timestep, stride):
      values = list(entries.values())
      times = list(entries.keys())
      entries = {}
      for index in range(0, len(values) - timestep + 1, stride):
            entries[times[index]] = values[index:index + timestep]
      return entries

def ToList(array):
      array = map(lambda x: list(x), array)
      return list(array)

def StandardizeArraySK(array, nValues):
      length = len(array)
      arrayNew = []

      if length <= nValues:
            scaler = StandardScaler().fit(array)
            standardized = scaler.transform(array)
            arrayNew = ToList(standardized)
            return arrayNew
      
      chunk = array[:nValues]
      scaler = StandardScaler().fit(chunk)
      standardized = scaler.transform(chunk)
      arrayNew += ToList(standardized)

      for index in range(nValues, length):
            chunk = array[index - nValues + 1:index + 1]
            scaler = StandardScaler().fit(chunk)
            standardized = scaler.transform([array[index]])
            arrayNew += ToList(standardized)

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

      
