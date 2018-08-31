import numpy as np
import matplotlib.pyplot as plt

# (1)向量简单卷积操作
# N = 5
# x = np.arange(5)
# weights = np.ones(N) / N
# print("Weights", weights)
# print(x)
# convolved_price = np.convolve(weights, x)
# print(convolved_price)

# (2)使用简单卷积操作对数据进行均化处理(Simple Moving Average)
# 一般取(N-1:-N+1)区间的元素，这些元素是x中的所有元素与weights做卷积操作后得到的
# N = 5
# weights = np.ones(N) / N
# print("Weights", weights)
# price = np.loadtxt('data/data.csv', delimiter=',', usecols=(6,),unpack=True)
# convolved_price = np.convolve(weights, price)
# valid_price = convolved_price[N-1:-N+1]
# t = np.arange(N - 1, len(price))            # 前N-1个元素是不完全卷积的结果，因此舍弃不用
# plt.plot(price, lw=1.0, label='Data')
# plt.plot(t, valid_price, '--', lw=2.0, label='Simple Moving Average')
# plt.title('5 Days Simple Moving Average')
# plt.xlabel('Days')
# plt.ylabel('Price ($)')
# plt.legend()
# plt.grid()
# plt.show()

# # (3)使用向量指数卷积操作对数据进行均化处理(Exponential Moving Average)
# N = 5
# weights = np.exp(np.linspace(-1., 0., N))
# weights /= weights.sum()
# print("Weights", weights)
# price = np.loadtxt('data/data.csv', delimiter=',', usecols=(6,),unpack=True)
# convolved_price = np.convolve(weights, price)[N-1:-N+1]
# t = np.arange(N - 1, len(price))            # 前N-1个元素是不完全卷积的结果，因此舍弃不用
# plt.plot(price, lw=1.0, label='Data')
# plt.plot(t, convolved_price, '--', lw=2.0, label='Exponential Moving Average')
# plt.title('5 Days Exponential Moving Average')
# plt.xlabel('Days')
# plt.ylabel('Price ($)')
# plt.legend()
# plt.grid()
# plt.show()

# (4)Bollinger Band:在SMA线的基础上，向上和向下各延申一段距离，从而形成所谓的"数据安全区域"
# 该距离的计算如下：原始数据点与SMA线上各点的残差平方和再开方，再乘以2
# N = 5
# weights = np.ones(N) / N
# print("Weights", weights)
# price = np.loadtxt('data/data.csv', delimiter=',', usecols=(6,), unpack=True)
# convolved_price = np.convolve(weights, price)[N-1:-N+1]
# deviation = []
# C = len(price)
# for i in range(N - 1, C):
#     if i + N < C:
#         dev = price[i: i + N]
#     else:
#         dev = price[-N:]
    
#     averages = np.zeros(N)
#     averages.fill(convolved_price[i - N - 1])       # 为何是-1而不是+1？
#     dev = dev - averages
#     dev = dev ** 2
#     dev = np.sqrt(np.mean(dev))
#     deviation.append(dev)
    
# deviation = 2 * np.array(deviation)
# print(len(deviation), len(convolved_price))
# upperBB = convolved_price + deviation
# lowerBB = convolved_price - deviation
# price_slice = price[N-1:]
# between_bands = np.where((price_slice < upperBB) & (price_slice > lowerBB))
# print(lowerBB[between_bands])
# print(price[between_bands])
# print(upperBB[between_bands])
# between_bands = len(np.ravel(between_bands))
# print("Ratio between bands", float(between_bands)/len(price_slice))
# t = np.arange(N - 1, C)
# plt.plot(t, price_slice, lw=1.0, label='Data')
# plt.plot(t, convolved_price, '--', lw=2.0, label='Moving Average')
# plt.plot(t, upperBB, '-.', lw=3.0, label='Upper Band')
# plt.plot(t, lowerBB, ':', lw=4.0, label='Lower Band')
# plt.title('Bollinger Bands')
# plt.xlabel('Days')
# plt.ylabel('Price ($)')
# plt.grid()
# plt.legend()
# plt.show()

