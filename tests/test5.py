import numpy as np
from statistics import NormalDist

a = (np.array([True, 0, 0, 1], dtype=int)
*
np.array([19.7, 2.4, 1.0, 3.2], dtype=float))

print(NormalDist(0, 1).inv_cdf(0.9))
print(0.999999 ** 2)