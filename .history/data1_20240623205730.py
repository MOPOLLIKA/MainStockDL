from data import FetchDataYF, FetchIndicators

def CommonKeyValues(dct1, dct2):
      keys1 = set(dct1.keys())
      keys2 = set(dct2.keys())
      commonKeys = sorted(list(keys1 & keys2))
      return dct


def TransformDataIntoSequence1(data, indicators):
      commonTime = CommonKeyValues(data, indicators)
      pass

TransformDataIntoSequence1(FetchDataYF(), FetchIndicators())