import numpy as np
import json
import random

a = {
  '1': 25,
  '2': 33,
  '3': 43
     }
data = json.dumps(a)

with open(f"/Users/MOPOLLIKA/python_StockDL/stockdata/test.json", "w") as f:
      json.dump(data, f, indent=2)
      f.close()

print(100 // 7)