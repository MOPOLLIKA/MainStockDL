from data import FetchDataYF, FetchIndicators

def CommonKeyValues(dct1, dct2):
      keys1 = set(dct1.keys())
      keys2 = set(dct2.keys())
      commonKeys = sorted(list(keys1 & keys2))
      dct1 = {time: dct1[time] for time in commonKeys}
      dct2 = {time: dct2[time] for time in commonKeys}
      return dct1, dct2

def TransformDataIntoSequence1(data, indicators):
      data, indicators = CommonKeyValues(data, indicators)
      for key in data:
            print()
      

TransformDataIntoSequence1(FetchDataYF(), FetchIndicators())