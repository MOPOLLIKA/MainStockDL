from data import FetchDataYF, FetchIndicators

print(FetchIndicators("IBM", "1d")[:100])
print(FetchDataYF("IBM", "1d")[:100])

