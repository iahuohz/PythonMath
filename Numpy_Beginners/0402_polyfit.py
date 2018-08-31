import numpy as np
import sys
import matplotlib.pyplot as plt

# (1)
num_origins = 100
x_origin = np.linspace(-np.pi, np.pi, num_origins)
y_origin = np.sin(x_origin) + np.random.randn(num_origins) * 0.1
poly = np.polyfit(x_origin, y_origin, 8)
y_pred = np.polyval(poly, x_origin)
plt.scatter(x_origin, y_origin)
plt.plot(x_origin, y_pred)
plt.show()

# (2)多项式拟合Polyfit
# bhp=np.loadtxt('data/BHP.csv', delimiter=',', usecols=(6,), unpack=True)
# vale=np.loadtxt('data/VALE.csv', delimiter=',', usecols=(6,), unpack=True)
# t = np.arange(len(bhp))
# poly = np.polyfit(t, bhp - vale, 3)
# print("Polynomial fit", poly)
# print("Next value", np.polyval(poly, t[-1] + 1))
# print("Roots", np.roots(poly))
# der = np.polyder(poly)
# print("Derivative", der)
# print("Extremas", np.roots(der))
# vals = np.polyval(poly, t)
# print(np.argmax(vals))
# print(np.argmin(vals))
# plt.plot(t, bhp - vale, label='BHP - VALE')
# plt.plot(t, vals, '--', label='Fit')
# plt.title('Polynomial fit')
# plt.xlabel('Days')
# plt.ylabel('Difference ($)')
# plt.grid()
# plt.legend()
# plt.show()