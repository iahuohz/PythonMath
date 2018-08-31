import datetime
import numpy as np

c = np.loadtxt('Numpy_Beginners/data/data.csv', delimiter=',', usecols=(6,), unpack=True)
# Initialize estimates array
estimates = np.zeros((len(c), 3))
for i in np.arange(len(c)):
    # Create a temporary copy and omit one value
    a = c.copy()
    a[i] = np.nan
    # Compute estimates
    estimates[i,] = [np.nanmean(a), np.nanvar(a), np.nanstd(a)]
print("Estimates variance", estimates.var(axis=0))


