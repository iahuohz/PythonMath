import numpy as np

# (1)线性拟合
N = 5
price = np.loadtxt('data/data.csv', delimiter=',', usecols=(6,), unpack=True)
slice_price = price[-N:]
slice_price = slice_price[::-1]
print("slice_price", slice_price)
A = np.zeros((N, N), float)         
for i in range(N):
    # A的每行是连续的5个点；一共5行数据，构成线性方程组的左侧矩阵
    # A中第1行的5个点，正好位于slice_price中第1个点的前面
    # 同理，第i行的5个点，正好位于slice_price中第i个点的前面
    # 这就相当于：根据前5个点的值，求第6个点的数据值
    A[i, ] = price[-N - 1 - i: - 1 - i]     
print("A", A)
(x, residuals, rank, s) = np.linalg.lstsq(A, slice_price, rcond=None)  # 求解最优的系数x
print(x, residuals, rank, s)
print(np.dot(slice_price, x))
print(np.dot(A[0], x))      # 计算第1行的预测结果，与slice_price中的第一个元素相同