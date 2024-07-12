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
      data, indicators = CommonKeyValues(data, indicators)
      data = {time: data[time][3] for time in data}
      volumes = {time}

if __name__ == "__main__":
      TransformDataIntoSequence1(FetchDataYF(), FetchIndicators())