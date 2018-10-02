import cupy as cp
import numpy as np
import math
import time

def DrawALine(point_1 = np.array((50,50)), point_2 = np.array((10,10))):
    if point_1[0] > point_2[0]:
        point_1, point_2 = point_2, point_1
    if abs(point_1[0] - point_2[0]) > abs(point_1[1] - point_2[1]):
        x = np.arange(point_1[0], point_2[0]+1)
        y = ((point_2[1] - point_1[1]) / (point_2[0] - point_1[0])) * (x - point_1[0]) + point_1[1]
        return np.array(x).astype(np.uint32), np.array(y).astype(np.uint32)
    else:
        x = np.arange(point_1[1], point_2[1]+1)
        y = ((point_2[0] - point_1[0]) / (point_2[1] - point_1[1])) * (x - point_1[1]) + point_1[0]
        return np.array(y).astype(np.uint32), np.array(x).astype(np.uint32)

def DrawALine_cuda(point_1 = np.array((50,50)), point_2 = np.array((10,10)), img = (100,100)):
    if point_1[0] > point_2[0]:
        point_1, point_2 = point_2, point_1
    if abs(point_1[0] - point_2[0]) > abs(point_1[1] - point_2[1]):
        x = cp.arange(point_1[0], point_2[0]+1)
        y = ((point_2[1] - point_1[1]) / (point_2[0] - point_1[0])) * (x - point_1[0]) + point_1[1]
        return cp.array(x).astype(cp.uint32), cp.array(y).astype(cp.uint32)
    else:
        x = cp.arange(point_1[1], point_2[1]+1)
        y = ((point_2[0] - point_1[0]) / (point_2[1] - point_1[1])) * (x - point_1[1]) + point_1[0]
        return cp.array(y).astype(cp.uint32), cp.array(x).astype(cp.uint32)

scale = 8.35

t_ = time.time()
DrawALine_cuda(np.array((0,0)), np.array((10**scale,10**scale)))
print(time.time() - t_)

t_ = time.time()
DrawALine(np.array((0,0)), np.array((10**scale,10**scale)))
print(time.time() - t_)
