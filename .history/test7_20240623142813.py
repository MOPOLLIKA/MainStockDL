import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd
import tensorflow as tf
from statistics import NormalDist

time = pd.Timestamp("2024-06-12")

a = np.array([np.array([195.3999939 , 196.8999939 , 194.86999512, 195.86999512,
       195.86999512,   0.22123841,   0.93905366,   0.43082003,
         0.64476799,  -0.58276766,   0.53271306,   0.74869252,
         0.59542574,   0.63710332,   0.95794345,  -0.20435957]), np.array([ 1.95690002e+02,  1.96500000e+02,  1.94169998e+02,  1.94479996e+02,
        1.94479996e+02, -6.48463799e-01,  8.71012274e-01, -4.51054982e-01,
        6.42997926e-01, -6.81200442e-01,  0.00000000e+00, -1.11880079e-01,
        3.30110664e-01,  5.74648445e-01,  7.35689011e-01, -4.27508318e-03]), np.array([ 1.94649994e+02,  1.96940002e+02,  1.94139999e+02,  1.96889999e+02,
        1.96889999e+02,  1.39950391e-01,  8.90027471e-01,  7.82323675e-01,
        6.60597306e-01, -6.50061258e-01,  4.65348015e-01,  5.41685203e-01,
        5.19117599e-01,  5.37696052e-01,  9.00985562e-01,  2.60453665e-03])])
b = np.array([np.array([ 1.94649994e+02,  1.96940002e+02,  1.94139999e+02,  1.96889999e+02,
        1.96889999e+02,  1.39950391e-01,  8.90027471e-01,  7.82323675e-01,
        6.60597306e-01, -6.50061258e-01,  4.65348015e-01,  5.41685203e-01,
        5.19117599e-01,  5.37696052e-01,  9.00985562e-01,  2.60453665e-03]), np.array([ 1.96899994e+02,  1.97300003e+02,  1.92149994e+02,  1.93119995e+02,
        1.93119995e+02,  9.97292822e-01,  7.07123195e-01, -6.57024715e-01,
        5.32018395e-01, -4.22736552e-01,  4.75319011e-01, -9.53380700e-01,
       -4.92344595e-01,  1.37588029e-01, -4.58441499e-01,  6.20219897e-01]), np.array([ 1.93649994e+02,  2.07160004e+02,  1.93630005e+02,  2.07149994e+02,
        2.07149994e+02,  9.99879492e-01,  9.23387135e-01,  9.98986432e-01,
        7.31205890e-01, -6.07292208e-01,  4.75319011e-01,  8.87092583e-01,
        3.41154724e-02,  2.44269514e-02,  9.99635961e-01, -7.82878244e-01])])

c = np.array(
[ [0.00069467],
  [0.00075686],
  [0.00078045],
  [0.00078942],
  [0.00079284],
  [0.00079415]])

if True: print("AAa")
/*
*
