import numpy as np
import json
from sklearn.preprocessing import normalize
from statistics import NormalDist
from data import StandardizeArrays1, StandardizeArrays2

a = [[1, 2, 3, 12, 4], [3, 4, 2, 8, 10], [3, 1, 34, 2, 5]]
#print(np.transpose(a))
#print(StandardizeArrays1(a, 3))
#print(StandardizeArrays2(a))
b = [[1, 2, 3, 4, 5]]
data = [{"avocado": [1, 2 ,3], "corn": [3, 2, 1], "pea": [3, 4, 9], "acapella": [1, 3, 12]}, {"bamboozle": 1, "money": 33}]
c = np.array([1, 2, 3])
d = np.array([])
vals = [a[0][3], a[1][3], a[2][3]]
std = np.std(vals)
mean = np.mean(vals)
# StandardizeArrays1 works weird!!
e = np.array([1, 2 ,3])
dct1 = {"a": 1, "b": 2}
dct2 = {"c": 3}
print(dct1.update(dct2))

with open("/Users/MOPOLLIKA/python_StockDL/test1.json", "w") as f:
      json.dump(data, f)
      f.close()