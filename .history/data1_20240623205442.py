from data import FetchDataYF, FetchIndicators

def CommonKeyValues(dct1, dct2):
      keys1 = set(dct1.keys())
      keys2 = set(dct2.keys())
      commonKeys = list(keys1 & keys2)
      print(commonKeys)
      return False


def TransformDataIntoSequence1(data, indicators):
      commonTime = CommonKeyValues(data, indicators)
      pass

TransformDAtaIntoSequence1(FetchDataYF("IBM", ))