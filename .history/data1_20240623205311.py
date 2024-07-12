from data import FetchDataYF, FetchIndicators

def CommonKeyValues(dct1, dct2):
      keys1 = set(dct1.keys())
      keys2 = set(dct2.keys())
      return keys1 & keys2


def TransformDataIntoSequence1(data, indicators):
      commonTime = CommonKeys(data, indicators)
      pass
