import requests 
import json
import random
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import datetime as dt
import tensorflow as tf
import yfinance
import keras
from statistics import NormalDist
from sklearn.preprocessing import normalize

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

def FetchDataYF(ticker="IBM", interval="1d"):
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

def FetchIndicators(ticker="IBM", interval="DAILY", series_type="close", apikey="U2NTP7T784ZP3OJK"):
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
      if not "Error Message" in indicators.keys():
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
  print(list(data.keys())[-1])
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

def LastWorkingDay

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

if __name__ == "__main__":
  ((x_train, y_train), (x_test, y_test)), (priceGrowths, (lastDateIndex, commonTime)) = CreateDataset(
  ticker="IBM",
  timeframe="DAILY",
  interval="1d",
  testFr=0.2,
  nValues=500,
  timestep=1,
  stride=1,
  )

  """for index in range(len(x_train)):
    print(f"X_{index}:")
    print(x_train[index])
    print(f"Y_{index}:")
    print(y_train[index])
  """

  
  