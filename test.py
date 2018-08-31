import datetime
import numpy as np

h, l, c = np.loadtxt('data/data.csv', delimiter=',', usecols=(4, 5, 6), unpack=True)
N = 5
h = h[-N:]
l = l[-N:]
print("len(h)", len(h), "len(l)", len(l))
print("Close", c)
previousclose = c[-N -1: -1]
print("len(previousclose)", len(previousclose))
print("Previous close", previousclose)
print("h - l ", h - l)
print("h - previousclose", h - previousclose)
print("previousclose - l", previousclose - l)
truerange = np.maximum(h - l, h - previousclose, previousclose - l)
print("True range", truerange)
atr = np.zeros(N)
atr[0] = np.mean(truerange)
for i in range(1, N):
    atr[i] = (N - 1) * atr[i - 1] + truerange[i]
    atr[i] /= N

print("ATR", atr)
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt


