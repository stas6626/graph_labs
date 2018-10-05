import cupy as cp
import numpy as np
import math
import time

center = np.array((2.3e8,2.3e8))
R = 2.3e8/2

t_ = time.time()
y = np.around(np.arange(-R/np.sqrt(2)-1,R/np.sqrt(2)+2,1)).astype(np.int32)
x = np.around((np.sqrt(R**2 - y**2))).astype(np.int32)
print(time.time() - t_)

t_ = time.time()
y = cp.arange(-R/np.sqrt(2)-1,R/np.sqrt(2)+2,1)
x = (cp.sqrt(R**2 - y**2)).astype(cp.int32)
print(time.time() - t_)
