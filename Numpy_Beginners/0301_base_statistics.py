import numpy as np
import matplotlib.pyplot as plt

# 基本统计函数
N = 10
x = np.arange(1, N+1)

# (1)平均数
print(np.mean(x), np.average(x))

# (2)加权平均数
# 时间加权平均数(Time-weighted average)
# 越是后面的数据，其权重也越大
v = np.arange(10)
print(np.average(x, weights=v), np.mean(np.dot(x, v / np.sum(v))))

# (3)极值差
print(np.ptp(x), np.max(x) - np.min(x))

# (4)中位数
x_sorted = np.sort(x)
print(np.median(x), (x_sorted[N // 2] + x_sorted[(N-1) // 2]) / 2)

# (5)方差和标准差
print(np.var(x), np.mean((x - np.mean(x)) ** 2))
print(np.std(x), np.sqrt(np.var(x)))

# (6)无偏方差和无偏标准差(样本方差、样本标准差)
print(np.var(x, ddof=1), (np.sum((x - np.mean(x)) ** 2) / (N - 1)))
print(np.std(x, ddof=1), np.sqrt(np.var(x, ddof=1)))

# (7)数组差分, 增长率
diff_arr = [x[i+1] - x[i] for i in np.arange(N - 1)]
print(np.diff(x), np.array(diff_arr))
diff_rate = np.diff(x) / x[:-1]
print(diff_rate)

# (8)对于两个向量或矩阵，选取对应位置元素的最大值，构成新数组返回
a1 = np.array([1,2,3,4,5])
a2 = np.array([0,1,5,3,7])
print(np.maximum(a1, a2))

# (9)数值截断
# 对于低于或高于制定阈值的元素，将其值设置为指定阈值
print("x=", x)
print("clipped=", x.clip(3, 6))

# (10)数据筛选
# 筛选出满足大小范围的数据，形成数组返回
print("x=", x)
print("compress=", x.compress(x>5))

# (11)依次计算从第一个元素到第i个元素的某种结果
print(x.cumsum())
print(x.cumprod())