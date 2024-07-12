import requests 
import json
import random
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import datetime as dt
import tensorflow as tf
import yfinance
import keras
from keras import layers
from sklearn.preprocessing import StandardScaler
from statistics import NormalDist
from sklearn.preprocessing import normalize
from csvtojson import CsvToEntries, AdjustForYears

def FetchDataAV(function="TIME_SERIES_DAILY", ticker="IBM", outputsize="full", interval="", apikey="U2NTP7T784ZP3OJK"):
      """
      Fetch indicator data using AlphaVantage
      """
      timeframe = function.split("_")[-1]
      name = ticker.lower() + timeframe.capitalize() + ".json"
      path = "/Users/MOPOLLIKA/python_StockDL/stockdata"
      if name in os.listdir(path):
            with open(os.path.join(path, name), "r") as f:
                  data = json.load(f)
                  f.close()
                  if not "Error Message" in data.keys():
                        return data

      url = "https://www.alphavantage.co/query?"
      if function:
            url += f"function={function}"
      if ticker:
            url += f"&symbol={ticker}"
      if outputsize:
            url += f"&outputsize={outputsize}"
      if interval:
            url += f"&interval{interval}"
      if apikey:
            url += f"&apikey={apikey}"
      r = requests.get(url)
      data = r.json()
      SaveData(data, timeframe=timeframe, ticker=ticker)
      return data

def FetchDataYF(ticker="AAPL", interval="1d"):
      """
      Fetch data via yahoo finance python library open, high, low, close, adj close, volume
      """
      data = yfinance.download(ticker, interval=interval)
      values = data.values
      times = data.index
      timeValues = {}
      for index in range(len(values)):
            timeValues[str(times[index].date())] = list(values[index])
      return timeValues

def unpackJSONIntoValues(data, name):
      return {f"{time}": list(data[f"Technical Analysis: {name}"][time].values()) for time in sorted(data[f"Technical Analysis: {name}"])}

def FetchIndicators(ticker="AAPL", interval="DAILY", series_type="close", apikey="U2NTP7T784ZP3OJK"):
      """
      Fetches powerful indicators in the following order:
      0. OBV
      1. AD
      2. ADX
      3. AROON
      4. RSI
      5. STOCH
      6. CCI
      7. MACD

      The data is passed as {'date': [OBV, AD, ... , MACD]} dictionary.
      """
      name = ticker.lower() + interval.capitalize() + "Indicators.json"
      path = "/Users/MOPOLLIKA/python_StockDL/stockdata"
      if name in os.listdir(path):
            with open(os.path.join(path, name), "r") as f:
                  indicators = json.load(f)
                  f.close()
                  if not "Error Message" in indicators.keys() and list(indicators.keys())[-1] in LastNumberWorkingDays(7):
                        print("Successfully loaded indicators from hard drive.")
                        return indicators
      # fetching all the required indicators data and transforming it into lists of values
      interval = interval.lower()
      ad = unpackJSONIntoValues(requests.get(f"https://www.alphavantage.co/query?function=AD&symbol={ticker}&interval={interval}&apikey={apikey}").json(), "Chaikin A/D")
      obv = ad  # unpackJSONIntoValues(requests.get(f"https://www.alphavantage.co/query?function=OBV&symbol={ticker}&interval={interval}&apikey={apikey}").json(), "OBV")
      adx = ad  # unpackJSONIntoValues(requests.get(f"https://www.alphavantage.co/query?function=ADX&symbol={ticker}&interval={interval}&time_period=14&apikey={apikey}").json(), "ADX")
      aroon = ad  # unpackJSONIntoValues(requests.get(f"https://www.alphavantage.co/query?function=AROON&symbol={ticker}&interval={interval}&time_period=25&apikey={apikey}").json(), "AROON")
      rsi = ad  # unpackJSONIntoValues(requests.get(f"https://www.alphavantage.co/query?function=RSI&symbol={ticker}&interval={interval}&time_period=14&series_type={series_type}&apikey={apikey}").json(), "RSI")
      stoch = ad  # unpackJSONIntoValues(requests.get(f"https://www.alphavantage.co/query?function=STOCH&symbol={ticker}&interval={interval}&apikey={apikey}").json(), "STOCH")
      cci = ad  # unpackJSONIntoValues(requests.get(f"https://www.alphavantage.co/query?function=CCI&symbol={ticker}&interval={interval}&time_period=20&apikey={apikey}").json(), "CCI")
      # finding macd via formula: macd = ema26 - ema12
      # ema26JSON = requests.get(f"https://www.alphavantage.co/query?function=EMA&symbol={ticker}&interval={interval}&time_period=26&series_type={series_type}&apikey={apikey}").json()
      # ema12JSON = requests.get(f"https://www.alphavantage.co/query?function=EMA&symbol={ticker}&interval={interval}&time_period=12&series_type={series_type}&apikey={apikey}").json()
      macd = ad # {f"{time}": [(float(ema26JSON["Technical Analysis: EMA"][time]["EMA"]) - float(ema12JSON["Technical Analysis: EMA"][time]["EMA"]))] for time in sorted(ema26JSON["Technical Analysis: EMA"])}
      # filling the indicators dictionary with values for different time
      indicators = {}
      commonTime = sorted(list((set(obv.keys()) & set(ad.keys()) & set(adx.keys()) & set(aroon.keys()) & set(rsi.keys()) & set(stoch.keys()) & set(cci.keys()) & set(macd.keys()))))
      for time in commonTime:
            values = list(map(lambda x: float(x), obv[time] + ad[time] + adx[time] + aroon[time] + rsi[time] + stoch[time] + cci[time] + macd[time]))
            indicators[time] = values
      SaveData(indicators, interval, ticker, indicators=True)
      return indicators


def SaveData(data, timeframe="DAILY", ticker="IBM", indicators=False):
      """
      Saves the price data to the stockdata folder as there is a limit for daily requests through the API.
      """
      indc = "Indicators" if indicators else ""
      name = ticker.lower() + timeframe.capitalize() + indc + ".json"
      try:
            with open(f"/Users/MOPOLLIKA/python_StockDL/stockdata/{name}", "w") as f:
                  json.dump(data, f)
                  f.close()
            return True
      except:
            print(f"Failed to create file with name: '{name}'")
            return False
      
def CreateDatasetFromDataSequence(inputValues, priceGrowths, testFr, timestep, stride):
      """
      Returns training and testing data from a sequence of multi-dimensional values
      """
      length = len(inputValues)
      random.seed(1387)
      growths = list(map(lambda x: x[1] / 10, priceGrowths)) # priceGrowths: list(tuple(price, growth, standardizedGrowth))
      inputEntries = [inputValues[index:index + timestep] for index in range(0, length - timestep, stride)]
      outputEntries = [growths[index:index + timestep] for index in range(0, length - timestep, stride)]
      lastDateIndex = range(0, length - timestep, stride)[-1]
      lengthSample = len(inputEntries)
      x_train = inputEntries[0:lengthSample - 1]
      y_train = outputEntries[1:lengthSample]
      testIndices = random.sample(range(0, lengthSample - 1), round(lengthSample * testFr))
      x_test, y_test = [], []
      for index in testIndices:
            x_test.append(x_train[index])
            y_test.append(y_train[index])
      return ((np.stack(x_train), np.stack(y_train)), (np.stack(x_test), np.stack(y_test))), lastDateIndex

def ToList(array):
      """For dewrapping a list of numpy single dimensianal arrays into a list of lists"""
      array = map(lambda x: list(x), array)
      return list(array)

def TensorToList(tensor: tf.Tensor):
      return tensor.numpy().tolist()

def StandardizeEntry(entry, entries, entryIndex, nValues):
      if entryIndex < nValues:
            entriesSlice = entries
      else:
            entriesSlice = entries[entryIndex - nValues + 1:entryIndex + 1]
      mean = np.average(entriesSlice)
      std = np.std(entriesSlice)
      if std != 0:
            return (NormalDist(mean, std).cdf(entry) - 0.5) * 2 # scaling the resulting cdf(0..1) to values (-1..1) where 0.5 -> 0, 0 -> -1, 1 -> 1
      else:
            return 1e-6
      
def StandardizeGrowth(growth, growths, growthIndex, nValues):
      if growthIndex < nValues:
            growthsSlice = growths
      else:
            growthsSlice = growths[growthIndex - nValues + 1:growthIndex + 1]
      growthsSlice = list(map(lambda x: abs(x), growthsSlice))
      std = np.mean(growthsSlice)
      if std == 0:
            return 1e-6
      value = NormalDist(0, std).cdf(growth)
      return (value - 0.5) * 2

      
def StandardizeArrays1(arrays, nValues):
      arraysNew = [[] for _ in range(len(arrays[0]))]
      valuesCurr = [[] for _ in range(len(arrays[0]))] 
      entryIndex = 0
      for array in arrays:
            for index in range(len(array)):
                  valuesCurr[index].append(array[index])
                  arraysNew[index].append(StandardizeEntry(array[index], valuesCurr[index], entryIndex, nValues))
            entryIndex += 1
      return ToList(np.transpose(arraysNew))

def StandardizeArrays2(arrays):
      return np.transpose(normalize(np.transpose(arrays)))

def CheckInputOutputMasks(inputMask, outputMask):
      """
      Checks whether inputted OHLCV flags of input correspond to the output flags
      """
      for index in range(len(inputMask)):
            if inputMask[index] == False and outputMask[index] == True:
                  return False
      return True

def TransformDataIntoSequence(data, indicators, frameKeyword, nValues, inputMask, outputMask, includeOHLC=True, includeVolume=False, YFoverAV=True): # nValues is the number of values to create a normal distribution for
      if not CheckInputOutputMasks(inputMask, outputMask):
            print("Value masks do not conform!")
            return 0
      if YFoverAV:
            if indicators:
                  commonTime = sorted(list(set(data.keys()) & set(indicators.keys())))
                  indicators = [list(indicators[time]) for time in commonTime]
            else: 
                  commonTime = list(data.keys())
            data = [data[time] for time in commonTime] 
            prices = list(map(lambda vals: vals[3], data))
            length = len(data)
      else:
            intradayInterval = data['Meta Data']['4. Interval'] if frameKeyword == "INTRADAY" else ""
            frames = {"DAILY": "Time Series (Daily)", "WEEKLY": "Weekly Time Series", "MONTHLY": "Monthly Time Series", "INTRADAY": f"Time Series ({intradayInterval})"}
            timeValues = data[frames[frameKeyword]]
            candles = [timeValues[time] for time in sorted(timeValues)]
            length = len(candles)
      inputValues = []
      inputMask = np.array(inputMask, dtype=int)
      outputMask = np.array(outputMask, dtype=int)
      volumes = []
      volumeInput = 0
      priceGrowths = []
      growths = []
      growthInput = 0
      """
      The volume problem solution idea is to create a standard(Gaussian) distribution where there is an accumulated mean and deviation
      """
      for entryIndex in range(length):
            if YFoverAV:
                  entryVolume = data[entryIndex][-1]
            else:
                  entryVolume = int(candles[entryIndex]["5. volume"])
            # volume input calculation
            volumes.append(entryVolume)
            volumeInput = StandardizeEntry(entryVolume, volumes, entryIndex, nValues)

            closePriceCurr = prices[entryIndex]
            if entryIndex != 0:
                  closePricePrev = prices[entryIndex - 1]
                  growth = (closePriceCurr - closePricePrev) / closePricePrev * 100
            else:
                  growth = 0
            growths.append(growth)
            growthInput = StandardizeGrowth(growth, growths, entryIndex, nValues)
            priceGrowths.append((closePriceCurr, growth, growthInput)) # price, growth, standardizedGrowth - tuples
            inputValue = np.array([growthInput] + [volumeInput], dtype=float)
            """
            else:
                  entry = candles[entryIndex]
                  entryData = np.array([entry["1. open"], entry["2. high"], entry["3. low"], entry["4. close"], volumeInput], dtype=float)
            # filtering out an input value
            inputValue = inputMask * entryData
            inputValue = inputValue[inputValue != 0]
            """
            # change inputValue from data to only input volume info with no other data
            if not includeOHLC:
                  inputValue = inputValue[-1]
            inputValue = np.append(inputValue, indicators[entryIndex]) if indicators else inputValue
            if not includeVolume:
                  inputValue = inputValue[0]
            inputValues.append(inputValue)
            """
            # filtering out an output value
            outputValue = outputMask * entryData
            outputValue = outputValue[outputValue != 0]
            outputValues.append(outputValue)
            """
      return inputValues, (priceGrowths, commonTime)

def CreateDataset(ticker="IBM", timeframe="DAILY", interval="1d", testFr=0.1, nValues=100, timestep=3, stride=2, 
                                                      inputMask=[False, False, False, True, False, True], outputMask=[False, False, False, True, False, False], includeIndicators=True, includeOHLC=True, includeVolume=True):
      data = FetchDataYF(ticker=ticker, interval=interval)
      if includeIndicators:
            indicators = FetchIndicators(ticker, timeframe)
            indicatorsValuesStandardized = StandardizeArrays1(list(indicators.values()), nValues)
            index = 0
            for time in indicators:
                  indicators[time] = indicatorsValuesStandardized[index]
                  index += 1
      else:
            indicators = []
      inputValues, (priceGrowths, commonTime) = TransformDataIntoSequence(data, indicators, timeframe, nValues, inputMask, outputMask, includeOHLC, includeVolume, YFoverAV=True)
      dataset, lastDateIndex = CreateDatasetFromDataSequence(inputValues, priceGrowths, testFr, timestep, stride) # (x_train, y_train), (x_test, y_test) - format
      return dataset, (priceGrowths, (lastDateIndex, commonTime))

def LastNumberWorkingDays(number):
      date = dt.date.today()
      timeDelta = dt.timedelta(1)
      dates = []

      while number != 0:
            while not IsTradingOnDate(date):
                  date -= timeDelta
            dates.append(str(date))
            date -= timeDelta
            number -= 1

      return dates

def NextWorkingDay(dateCurrent):
      timeDifference = dt.timedelta(1)
      dateNext = dateCurrent + timeDifference
      while not IsTradingOnDate(dateNext):
            dateNext += timeDifference
      return dateNext

def DaysPerEntry(interval):
      match interval.lower():
            case "daily":
                  return 1
            case "weekly":
                  return 7
            case "monthly":
                  return 30
            case _:
                  raise ValueError("Interval is not in {'daily', 'weekly', 'monthly'}.")

def PredictEntries(model: keras.Model, n, commonTime, lastDateIndex, lastEntry, interval, timestep, stride):
      valuesPredicted = dict()
      lastDate = commonTime[lastDateIndex + timestep - stride + 1]
      dateZero = dt.datetime.strptime(lastDate, "%Y-%m-%d")
      daysPerCandle = dt.timedelta(DaysPerEntry(interval))

      dates = []
      index = n
      u = 0
      while index != 0:
            date = (dateZero + daysPerCandle * (u + 1)).date()
            if IsTradingOnDate(date):
                  dates.append(str(date))
                  index -= 1
            u += 1
      
      nextEntry = lastEntry
      i = 0
      while i <= 5 * n:
            nextEntry = model(np.array(nextEntry).reshape((-1, 1)))
            dateZero = dateZero + (stride * daysPerCandle)
            for innerIndex in range(timestep):
                  date = dateZero + (innerIndex * daysPerCandle)
                  if date not in commonTime: 
                        valuesPredicted[str(date.date())] = nextEntry[innerIndex]
                        i += 1
      valuesPredicted = {date: valuesPredicted[date] for date in valuesPredicted if date in dates}
      return valuesPredicted

def IsTradingOnDate(date: dt.datetime):
      dayOfWeek = date.weekday()
      if dayOfWeek in {5, 6}: # is it Saturday or Sunday
            return False
      return True

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

def PredictValuesForAnyValues(numberOfValuesToPredict, model: keras.Model, entries, numberOfValues):
      """
      Predicts a certain number of candles ahead in the future using a model and entries data with clusters of candles. It returns {"date": value, "date": value, ...}
      """
      #entriesTransformed = TransformEntries(entries, number, number)
      valuesPredicted = {}
      predictionInput = np.reshape(list(entries.values())[-numberOfValuesToPredict:], (1, numberOfValuesToPredict, numberOfValues))
      predictionOutput = model(predictionInput)
      predictionOutput = TensorToList(predictionOutput)

      timeDifference = TimeDifference(entries)
      dateLast = dt.datetime.strptime(list(entries.keys())[-1], "%Y-%m-%d")
      dateNext = dateLast + timeDifference
      for index in range(numberOfValuesToPredict):
            while not IsTradingOnDate(dateNext):
                  dateNext += timeDifference
            valuesPredicted[str(dateNext.date())] = predictionOutput[0][index]
            dateNext += timeDifference
      return valuesPredicted

def PredictValuesFor1Values(numberOfValuesToPredict, model: keras.Model, entries, timestep):
      """
      Predicts a certain number of candles ahead in the future using a candle by candle predicted method.
      """
      valuesPredicted = {}
      predictions = []
      predictionInput = np.reshape(list(entries.values())[-timestep:], (1, timestep, 1))

      dateLast = dt.datetime.strptime(list(entries.keys())[-1], "%Y-%m-%d")
      dateNext = NextWorkingDay(dateLast)
      for _ in range(numberOfValuesToPredict):
            predictionOutput = model(predictionInput)
            prediction = TensorToList(predictionOutput)
            predictions.append(prediction)
            valuesPredicted[str(dateNext.date())] = prediction[0][-1]
            dateNext = NextWorkingDay(dateNext)
            predictionInput = np.reshape(predictions[-1][0], (1, timestep, 1))

      return valuesPredicted

def TransformDataIntoSequence1(data, indicators, numberOfValues=3):
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
      volumes = DatesToValues(dates, volumesStandardized1)

      indicatorsValues = {time: indicators[time][1] for time in data} # take only certain indicator values
      indicatorsListed = list(map(lambda x: list(x), list(np.reshape(list(indicatorsValues.values()), (-1, 1)))))
      indicatorsStandardized1 = StandardizeArrays1(indicatorsListed, nValues)
      indicatorsStandardized2 = StandardizeArraySK(indicatorsListed, nValues)
      indicators = DatesToValues(dates, indicatorsStandardized1)

      if numberOfValues == 3:
            entries = {time: [closes[time]] + volumes[time] + indicators[time] for time in data}
      elif numberOfValues == 1:
            entries = {time: [closes[time]] for time in data}
      else:
            print("Either 3 or 1 is accepted")
            return False
      return entries

def TransformEntries(entries, timestep, stride):
      values = list(entries.values())
      times = list(entries.keys())
      entries = {}
      for index in range(0, len(values) - timestep + 1, stride):
            entries[times[index]] = values[index:index + timestep]
      return entries

def TransformEntriesFrom3To1(entries):
      entries = {time: entries[time][0] for time in entries}
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

def CreateDatasetFromEntries(entries, numberOfValues):
      x_train = np.stack(np.array([entries[time] for time in list(entries.keys())[0:-1]]))
      y_train = []
      for time in list(entries.keys())[1:]:
            entry = np.reshape(entries[time], (-1, numberOfValues))
            y_train.append([[array[0]] for array in entry]) # made a change from: y_train.append([[array[0] for array in entry]])

      y_train = np.stack(y_train)
      return x_train, y_train
      
def TestModel(model: keras.Model, x_train, y_train):
      print(x_train[-2:])
      print(model(x_train[-2:]))
      print(y_train[-2:])
      return True

def TestDataset(x_train, y_train):
      if len(x_train) != len(y_train):
            print("The datasets are of different sizes!")
            return False
      for index in range(len(x_train)):
            print(f"x_train{index}: {x_train[index]}, y_train{index}: {y_train[index]}")
      return True

def PlotPredictionGraphFor3Values(model: keras.Model, x_train, y_train, entries, predictionUncovered):
      entriesCloseValues = TransformEntriesFrom3To1(entries)
      numberOfPoints = len(entries)
      xActual = range(1, numberOfPoints + 1)
      yActual = entriesCloseValues
      xPredicted = range(1, numberOfPoints + 1)

      yPredicted = []
      #yPredicted += list(np.ravel(x_train[0]))
      for entryTrain in x_train:
            modelInput = np.reshape(entryTrain, (1, -1, 3))
            prediction = TensorToList(model(modelInput))
            yPredicted += list(np.ravel(prediction))     
      #yPredicted += list(np.ravel(y_train[-1]))

      timestep = len(x_train[0])
      numberLeft = numberOfPoints % timestep
      #yPredicted += list(np.ravel(list(entries.values())[-timestep:]))
      sequenceLeft = list(entries.values())[-2 * timestep:-timestep]
      inputLeft = np.reshape(sequenceLeft, (1, timestep, 3)) # (1, timestep, 1)
      outputLeft = TensorToList(model(inputLeft))
      yPredicted += list(np.ravel(outputLeft)[-numberLeft:])
      
      xUncovered = range(numberOfPoints + 1, len(predictionUncovered) + numberOfPoints + 1)
      yUncovered = list(map(lambda x: x[0], predictionUncovered.values()))

      plt.plot(xActual, yActual, label="Actual", color="red")
      plt.plot(xPredicted, yPredicted, label="Covered prediction", color="blue")
      plt.plot(xUncovered, yUncovered, label="Uncovered prediction", color="green", linestyle="-", marker=".", markersize=5)
      plt.plot([xPredicted[-1], xUncovered[0]], [yPredicted[-1], yUncovered[0]], color="green", linestyle="-")
      plt.legend(loc="upper left")
      plt.show()
      return True

def PlotPredictionGraphFor1Values(model: keras.Model, x_train, entries):
      entries = TransformEntriesFrom3To1(entries)
      values = list(entries.values())
      numberOfPoints = len(values)
      xActual = range(1, numberOfPoints + 1)
      yActual = values
      xPredicted = range(1 + timestep, numberOfPoints + 1)

      yPredicted = []
      for predictionInput in x_train:
            predictionOutput = model(predictionInput)
            predictionOutput = TensorToList(predictionOutput)
            prediction = predictionOutput[0][-1] # [0]
            yPredicted.append(prediction)
      
      plt.plot(xActual, yActual, color="red", label="Actual")
      plt.plot(xPredicted, yPredicted, color="blue", label="Covered prediction")
      plt.show()
      return True

      
def CreateBasicModel(inputShape, units):
      model = keras.Sequential()
      model.add(keras.Input(shape=inputShape))
      model.add(layers.Dense(units))
      model.add(layers.Dropout(0.1))
      model.add(layers.Dense(1))

      model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-3),
            loss=keras.losses.MeanAbsolutePercentageError()
      )
      #model.summary()
      return model

def CreateLstmModel(inputShape, units):
      model = keras.Sequential()
      model.add(keras.Input(shape=inputShape))
      model.add(layers.LSTM(units, return_sequences=True))
      model.add(layers.LSTM(units, return_sequences=False))
      model.add(layers.Dense(units // 2))
      model.add(layers.Dense(1))

      model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-4),
            loss=keras.losses.MeanAbsolutePercentageError()
      )
      model.summary()
      return model

# I am not good enough in convnets yet
def CreateConvolutionalModel(inputShape, filters, kernelSize):
      inputs = keras.Input(shape=inputShape)
      x = layers.Conv1D(filters, kernelSize)(inputs)
      x = layers.MaxPool1D()(x)
      x = layers.Dense(32)(x)
      outputs = layers.Dense(1)(x)
      return keras.Model(inputs=inputs, outputs=outputs)

def SaveStocksEntries(stocksEntries):
      with open("/Users/MOPOLLIKA/python_StockDL/stockdata/stocksEntries.json", "w") as f:
            json.dump(stocksEntries, f)
            f.close()
      return True

def GetStocksEntries(numberOfValues=3):
      tickers = ["MSFT", "AAPL", "NVDA", "AVGO", "AMZN" ,"META", "TSLA", "GOOGL", "GOOG", "COST", 
                 "NFLX", "AMD", "ADBE", "QCOM", "PEP", "TMUS", "LIN", "AMAT", "CSCO", "INTU", 
                 "TXN", "AMGN", "ISRG", "CMCSA", "MU", "LRCX", "HON", "INTC", "BKNG", "VRTX", 
                 "KLAC", "ADI", "REGN", "PANW", "ASML", "ADP", "SNPS", "PDD", "CRWD", "MDLZ", 
                 "CDNS", "SBUX", "MELI", "GILD", "CTAS", "NXPI", "MAR", "ABNB", "CEG", "CSX", 
                 "PYPL", "MRVL", "ORLY", "ROP", "ADSK", "PCAR", "CRPT", "MNST", "MCHP", "ROST", 
                 "WDAY", "FTNT", "AEP", "MRNA", "DXCM", "AZN", "TTD", "KDP", "CHTR", "PAYX", 
                 "DASH", "DDOG", "IDXX", "ODFL", "VRSK", "KHC", "EA", "FANG", "FAST", "LULU", 
                 "GEHC", "EXC", "BKR", "CCEP", "CTSH", "BIIB", "ON", "ZS", "TEAM", "CSGP", 
                 "GFS", "XEL", "CDW", "ANSS", "TTWO", "DLTR", "ARM", "MDB", "WBD", "ILMN"]
      with open("/Users/MOPOLLIKA/python_StockDL/stockdata/stocksEntries.json", "r") as f:
            stocksEntriesCurrent = json.load(f)
            f.close()

      # Checks whether there are requests left
      ad = requests.get(f"https://www.alphavantage.co/query?function=AD&symbol=IBM&interval=DAILY&apikey=U2NTP7T784ZP3OJK").json()
      if "Information" in ad.keys():
            print("No more AV requests left. Fetching available data...")
            return stocksEntriesCurrent
      
      # Checks whether the data is complete and no fetch is required
      if "ILMN" in stocksEntriesCurrent.keys():
            entriesILMN = stocksEntriesCurrent["ILMN"]
            if list(entriesILMN.keys())[-1] in LastNumberWorkingDays(5):
                  print(f"Downloaded stocks data from the memory.")
                  return stocksEntriesCurrent

      stocksEntries = {}
      for ticker in tickers:
            data = FetchDataYF(ticker=ticker, interval="1d")
            try:
                  indicators = FetchIndicators(ticker=ticker, interval="DAILY")
            except:
                  SaveStocksEntries(stocksEntries)
                  print(f"Number of indicator requests has exceeded the limit of AlphaVantage. The last ticker loaded is: {list(stocksEntries.keys())[-1]}.") if stocksEntries else \
                  print(f"No entries fetched due to the limit exceeded")
                  return stocksEntries
            entries = TransformDataIntoSequence1(data, indicators, numberOfValues=numberOfValues)
            stocksEntries[ticker] = entries
            print(f"Fetched entries for \033[1m{ticker}\033[0m") # \033[1m...\033[0m makes the text bold

      SaveStocksEntries(stocksEntries)
      return stocksEntries

def TrainBaseNumberModel(numberOfTickers, numberOfValues, timestep, lstm=False):
      stocksEntries = GetStocksEntries(numberOfValues=numberOfValues)
      model = CreateBasicModel((timestep, numberOfValues), 128) if lstm == False else CreateLstmModel((timestep, numberOfValues), 64)
      checkpoints = keras.callbacks.ModelCheckpoint(filepath=f"/Users/MOPOLLIKA/python_StockDL/ModelCheckpoints/base{numberOfTickers}_{numberOfValues}{'_lstm' if lstm else ''}.weights.h5", save_weights_only=True, verbose=1)
      earlyStopping = keras.callbacks.EarlyStopping("val_loss", patience=5)
      stride = timestep if numberOfValues == 3 else 1
      for ticker in list(stocksEntries.keys())[0:numberOfTickers]:
            entries = stocksEntries[ticker]
            if numberOfValues == 1:
                  entries = TransformEntriesFrom3To1(entries)
            entriesTransformed = TransformEntries(entries, timestep, stride)
            x_train, y_train = CreateDatasetFromEntries(entriesTransformed, numberOfValues)
            model.fit(x_train, y_train, batch_size=1, epochs=1, verbose=1, callbacks=checkpoints, validation_split=0.1)
      return model

def CreateEntries(ticker, numberOfValues):
      data = FetchDataYF(ticker=ticker)
      indicators = FetchIndicators(ticker=ticker)
      entries = TransformDataIntoSequence1(data, indicators, numberOfValues)
      return entries

def CreateDataset1(entries, timestep, numberOfValues):
      stride = timestep if numberOfValues == 3 else 1
      entriesTransformed = TransformEntries(entries, timestep, stride)
      x_train, y_train = CreateDatasetFromEntries(entriesTransformed, numberOfValues)
      return x_train, y_train

def LoadModelBaseNumberBasic(timestep, numberOfValues, numberOfTickers):
      path = f"/Users/MOPOLLIKA/python_StockDL/ModelCheckpoints/base{numberOfTickers}_{numberOfValues}.weights.h5"
      modelBase = CreateBasicModel((timestep, numberOfValues), 128)
      modelBase.load_weights(path)
      return modelBase

def LoadModelBaseNumberLstm(timestep, numberOfValues, numberOfTickers) -> keras.Model:
      path = f"/Users/MOPOLLIKA/python_StockDL/ModelCheckpoints/base{numberOfTickers}_{numberOfValues}_lstm.weights.h5"
      modelBase = CreateLstmModel((timestep, numberOfValues), 64)
      modelBase.load_weights(path)
      return modelBase

def IsBetterThanRandom(model: keras.Model, x_test, y_test) -> bool:
      result = model.evaluate(x_test, y_test)
      
      modelRandom = ModelRandom()
      resultRandom = modelRandom.evaluate(x_test, y_test)

      if result < resultRandom:
            return True
      else:
            return False

class ModelEnsemble(keras.Model):
      def __init__(self, model, modelBase):
            super().__init__()
            self.model = model
            self.modelBase = modelBase
      
      def __call__(self, inputs):
            modelOutput = self.model(inputs)
            modelBaseOutput = self.modelBase(inputs)
            ensembleOutput = (modelOutput + modelBaseOutput) / 2
            return ensembleOutput

class ModelRandom(keras.Model):
      def __init__(self):
            super().__init__()
            self.compile(optimizer=keras.optimizers.Adam(),
                         loss=keras.losses.MeanAbsolutePercentageError())

      def __call__(self, inputs) -> tf.Tensor:
            inputs = tf.convert_to_tensor(inputs)
            return inputs
      
if __name__ == "__main__":
      timestep = 1
      numberOfValues = 1
      numberOfValuesToPredict = 5
      numberOfTickers = 20

      #TrainBaseNumberModel(numberOfTickers, numberOfValues, timestep, lstm=True)
      #entries = CreateEntries("NFLX", numberOfValues)
      entries: dict = CsvToEntries("/Users/MOPOLLIKA/python_StockDL/globaltemperature/GlobalLandTemperatures_GlobalTemperatures.csv")
      entries: dict = AdjustForYears(entries)
      x_train, y_train = CreateDataset1(entries, timestep, numberOfValues)
      """
      model = CreateBasicModel((timestep, numberOfValues), 64)
      earlyStopping = keras.callbacks.EarlyStopping("val_loss", patience=5, restore_best_weights=True)
      model.fit(x_train, y_train, batch_size=1, epochs=50, verbose=1, callbacks=earlyStopping, validation_split=0.1)
      """
      modelBase = LoadModelBaseNumberLstm(timestep, numberOfValues, numberOfTickers)
      modelBase.fit(x_train, y_train, batch_size=1, epochs=1, validation_split=0.1)

      model = CreateBasicModel((timestep, numberOfValues), 128)
      model.fit(x_train, y_train, batch_size=1, epochs=1, validation_split=0.1)

      #modelEnsemble = ModelEnsemble(model, modelBase)

      #prediction = PredictValuesFor1Values(numberOfValuesToPredict, modelEnsemble, entries, timestep)
      
      #PlotPredictionGraphFor1Values(model, x_train, entries)
      print(IsBetterThanRandom(modelBase, x_train, y_train))


      """   

      stocksEntries = GetStocksEntries(numberOfValues)
      results = []
      for ticker in stocksEntries:
            entries = stocksEntries[ticker]
            if numberOfValues == 1:
                  entries = TransformEntriesFrom3To1(entries)
            x_train, y_train = CreateDataset1(entries, timestep, numberOfValues)
            model = CreateBasicModel((timestep, numberOfValues), 128)

            #modelBase = CreateBasicModel((timestep, numberOfValues), 128)
            #modelBase.load_weights(f"/Users/MOPOLLIKA/python_StockDL/ModelCheckpoints/base{numberOfTickers}_{numberOfValues}.weights.h5")

            earlyStopping = keras.callbacks.EarlyStopping("val_loss", patience=5, restore_best_weights=True)
            model.fit(x_train, y_train, batch_size=1, epochs=50, callbacks=earlyStopping, validation_split=0.1, verbose=0)
            result = model.evaluate(x_train, y_train, batch_size=1, verbose=2)
            results.append(result)
      for result in results:
            print(result)

            
      PlotPredictionGraphFor3Values(model, x_train, y_train, entries, prediction)
      """