import numpy as np
import matplotlib.pyplot as plt

# (1)线性拟合
# N = 5
# price = np.loadtxt('data/data.csv', delimiter=',', usecols=(6,), unpack=True)
# slice_price = price[-N:]
# slice_price = slice_price[::-1]
# print("slice_price", slice_price)
# A = np.zeros((N, N), float)         
# for i in range(N):
#     # A的每行是连续的5个点；一共5行数据，构成线性方程组的左侧矩阵
#     # A中第1行的5个点，正好位于slice_price中第1个点的前面
#     # 同理，第i行的5个点，正好位于slice_price中第i个点的前面
#     # 这就相当于：根据前5个点的值，求第6个点的数据值
#     A[i, ] = price[-N - 1 - i: - 1 - i]     
# print("A", A)
# (x, residuals, rank, s) = np.linalg.lstsq(A, slice_price, rcond=None)  # 求解最优的系数x
# print(x, residuals, rank, s)
# print(np.dot(slice_price, x))
# print(np.dot(A[0], x))      # 计算第1行的预测结果，与slice_price中的第一个元素相同

# (2)利用线性拟合做出趋势图
def fit_line(t, y):
    ''' Fits t to a line y = at + b '''
    A = np.vstack([t, np.ones_like(t)]).T
    return np.linalg.lstsq(A, y)[0]

# Determine pivots
# h:最高价 l:最低价 c:收盘价
h, l, c = np.loadtxt('data/data.csv', delimiter=',', usecols=(4, 5, 6), unpack=True)
pivots = (h + l + c) / 3
print("Pivots", pivots)

# Fit trend lines
t = np.arange(len(c))
sa, sb = fit_line(t, pivots - (h - l))      # support levels
ra, rb = fit_line(t, pivots + (h - l))      # resistance levels
support = sa * t + sb
resistance = ra * t + rb
condition = (c > support) & (c < resistance)
print("Condition", condition)
between_bands = np.where(condition)
print(support[between_bands])
print(c[between_bands])
print(resistance[between_bands])
between_bands = len(np.ravel(between_bands))
print("Number points between bands", between_bands)
print("Ratio between bands", float(between_bands)/len(c))
print("Tomorrows support", sa * (t[-1] + 1) + sb)
print("Tomorrows resistance", ra * (t[-1] + 1) + rb)
a1 = c[c > support]
a2 = c[c < resistance]
print("Number of points between bands 2nd approach" ,len(np.
intersect1d(a1, a2)))
# Plotting
plt.plot(t, c, label='Data')
plt.plot(t, support, '--', lw=2.0, label='Support')
plt.plot(t, resistance, '-.', lw=3.0, label='Resistance')
plt.title('Trend Lines')
plt.xlabel('Days')
plt.ylabel('Price ($)')
plt.grid()
plt.legend()
plt.show()