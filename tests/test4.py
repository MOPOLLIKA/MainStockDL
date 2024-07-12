import os
import requests
import json

name = "ibmweekly.json"
path = "/Users/MOPOLLIKA/python_StockDL/stockdata"
pathname = os.path.join(path, name)
with open(pathname, "r") as f:
  data = json.load(f)
  f.close()
url = 'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol=IBM&interval=30min&apikey=demo'
r = requests.get(url)
data = r.json()

print(data)