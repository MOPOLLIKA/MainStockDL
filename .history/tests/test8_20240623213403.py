import datetime as dt
import numpy as np
from  import PredictEntries
import keras

#PredictEntries(keras.Model(), 10, 1, "2024-06-17", 1, "daily", 1, 1)
a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 0])
b = np.array([2, 3, 1, 7, 9, 0 ,4 ,5, 6, 8])
print(b[0:-1])