import keras
import numpy as np
import datetime as dt
from keras import layers
from data import FetchDataYF, FetchIndicators, StandardizeArrays1, ToList, IsTradingOnDate
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

def TimeDifference(entries):
      """
      Finds the difference in time between candles in a given data sequence
      """
      datesLast = list(entries.keys())[-5:]
      datesLastDT = list(map(lambda x: dt.datetime.strptime(x, "%Y-%m-%d"), datesLast))
      datesDifference = [datesLastDT[index] - datesLastDT[index - 1] for index in range(1, len(datesLastDT))]
      return min(datesDifference)

def PredictValues(number, model: keras.Model, entries):
      """
      Predicts a certain number of candles ahead in the future using a model and entries data. It returns {"date": value, "date": value, ...}
      """
      entriesTransformed = TransformEntries(entries, number, number)
      x_train, y_train = CreateDatasetFromEntries(entriesTransformed)

      earlyStopping = keras.callbacks.EarlyStopping("val_loss", patience=5, restore_best_weights=True)
      model.fit(x_train, y_train, batch_size=4, epochs=50, verbose=1, callbacks=earlyStopping, validation_split=0.2)

      valuesPredicted = {}
      predictionInput = np.array(list(entriesTransformed.values())[-1:])
      predictionOutput = ToList(model(predictionInpu
      timeDifference = TimeDifference(entries)
      print(f"output: {predictionOutput}")

      dateLast = dt.datetime.strptime(list(entries.keys())[-1], "%Y-%m-%d")
      dateNext = dateLast + timeDifference
      for index in range(number):
            while not IsTradingOnDate(dateNext):
                  dateNext += timeDifference
            valuesPredicted[str(dateNext.date())] = predictionOutput[0][index]
            dateNext += timeDifference
      return valuesPredicted

def TransformDataIntoSequence1(data, indicators):
      """
      Simple transformation of data into a sequence of [close, volume, indicator] values attached to time keys in a dictionary.
      """
      data, indicators = CommonKeyValues(data, indicators)
      dates = list(data.keys())
      nValues = 500

      closes = {time: data[time][3] for time in data}

      volumesValues = {time: data[time][3] for time in data}
      volumesListed = list(map(lambda x: list(x), list(np.reshape(list(volumesValues.values()), (-1, 1)))))
      volumesStandardized1 = StandardizeArrays1(volumesListed, nValues)
      volumesStandardized2 = StandardizeArraySK(volumesListed, nValues)
      volumes = DatesToValues(dates, volumesStandardized2)

      indicatorsValues = {time: indicators[time][1] for time in data} # take only certain indicator values
      indicatorsListed = list(map(lambda x: list(x), list(np.reshape(list(indicatorsValues.values()), (-1, 1)))))
      indicatorsStandardized1 = StandardizeArrays1(indicatorsListed, nValues)
      indicatorsStandardized2 = StandardizeArraySK(indicatorsListed, nValues)
      indicators = DatesToValues(dates, indicatorsStandardized2)

      entries = {time: [closes[time]] + volumes[time] + indicators[time] for time in data}
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

      model = keras.Sequential()
      model.add(keras.Input(shape=(None, 3, )))
      model.add(layers.Dense(64))
      model.add(layers.Dense(1))

      model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-5), loss=keras.losses.MeanSquaredError())
      model.summary()
      prediction = PredictValues(3, model, entries)
      print(prediction)

      """
      earlyStopping = keras.callbacks.EarlyStopping("val_loss", patience=5, restore_best_weights=True)
      model.fit(x_train, y_train, batch_size=4, epochs=50, callbacks=earlyStopping, verbose=1, validation_split=0.2)

      TestModel(model, x_train, y_train)
      """

      