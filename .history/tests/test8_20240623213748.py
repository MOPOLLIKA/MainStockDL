import datetime as dt
import numpy as np
import sys
import keras
sys.path.insert(0, "/Users/MOPOLLIKA/python_StockDL/data.py")
from data import PredictEntries
#PredictEntries(keras.Model(), 10, 1, "2024-06-17", 1, "daily", 1, 1)
a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 0])
b = np.array([2, 3, 1, 7, 9, 0 ,4 ,5, 6, 8])
print(b[0:-1])