import datetime as dt
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from data import LastNumberWorkingDays 

#PredictEntries(keras.Model(), 10, 1, "2024-06-17", 1, "daily", 1, 1)
a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 0])
b = np.array([2, 3, 1, 7, 9, 0 ,4 ,5, 6, 8])
c = np.array([[[1.65279999e+02, 2.77610000e+06, 3.94577159e+08], [1.65809998e+02, 2.59420000e+06, 3.97171362e+08], [1.67380005e+02, 3.04940000e+06, 4.00220739e+08]],[[1.67380005e+02, 3.04940000e+06, 4.00220739e+08], [1.68199997e+02, 2.20710000e+06, 4.02428002e+08], [1.70009995e+02, 3.47550000e+06, 4.05903497e+08]]])
d = {"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]}

output = np.array([[454.63983], [456.74887]])
output = []
print(output)

print(type(LastNumberWorkingDays(10)[0]))

