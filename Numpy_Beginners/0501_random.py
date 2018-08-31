import numpy as np
import matplotlib.pyplot as plt

# # (1) 二项分布
# cash = np.zeros(10000)
# cash[0] = 1000
# np.random.seed(73)
# outcome = np.random.binomial(9, 0.5, size=len(cash))
# for i in range(1, len(cash)):
#     if outcome[i] < 5:
#         cash[i] = cash[i - 1] - 1
#     elif outcome[i] < 10:
#         cash[i] = cash[i - 1] + 1
#     else:
#         raise AssertionError("Unexpected outcome " + outcome)
# print(outcome.min(), outcome.max())
# plt.plot(np.arange(len(cash)), cash)
# plt.title('Binomial simulation')
# plt.xlabel('# Bets')
# plt.ylabel('Cash')
# plt.grid()
# plt.show()

# (2)超几何分布
# points = np.zeros(100)
# np.random.seed(16)
# outcomes = np.random.hypergeometric(25, 1, 3, size=len(points))
# for i in range(len(points)):
#     if outcomes[i] == 3:
#         points[i] = points[i - 1] + 1
#     elif outcomes[i] == 2:
#         points[i] = points[i - 1] - 6
#     else:
#         print(outcomes[i])
# plt.plot(np.arange(len(points)), points)
# plt.title('Game show simulation')
# plt.xlabel('# Rounds')
# plt.ylabel('Score')
# plt.grid()
# plt.show()

# (3)正态分布
# N=10000
# np.random.seed(27)
# normal_values = np.random.normal(size=N)
# _, bins, _ = plt.hist(normal_values, int(np.sqrt(N)), normed=True, lw=1, label="Histogram")
# sigma = 1
# mu = 0
# plt.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) * np.exp( - (bins -mu)**2 / (2 * sigma**2) ), 
#     '--', lw=3, lael="PDF")
# plt.title('Normal distribution')
# plt.xlabel('Value')
# plt.ylabel('Normalized Frequency')
# plt.grid()
# plt.legend(loc='best')
# plt.show()

# (4) 随机筛选 choice
N = 500
data = np.random.normal(size=N)
print(data.mean())
bootstrapped = np.random.choice(data, size=(N, 100))
means = bootstrapped.mean(axis=0)
print(means.mean())