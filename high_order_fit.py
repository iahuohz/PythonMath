import numpy as np
import scipy.optimize as opt
from matplotlib import pyplot as plt

# 定义拟合函数
def fit_function(W, X):
    return np.dot(X, W)

# 定义残差函数
def error_function(W, X, y):
    return fit_function(W, X) - y

def make_x_ext(x_org, order):
    x_ext = np.c_[x_org[:,np.newaxis], np.ones(len(x_org))]            # 追加全1列
    for i in np.arange(1, order):                                      # 依次构造各阶数据
        x_ext = np.insert(x_ext, 0, np.power(x_org, i), axis=1)
    return x_ext

num_origins = 10
x_origin = np.linspace(-np.pi, np.pi, num_origins)
y_origin = np.sin(x_origin) + np.random.randn(num_origins) * 0.1
num_predicts = 100
x_predict = np.linspace(-np.pi, np.pi, num_predicts)

min_order = 3
max_order = 9
step = 2

for order in np.arange(min_order, max_order+1, step):
    w_init = np.random.rand(order+1)
    x_ext = make_x_ext(x_origin, order)
    result = opt.leastsq(error_function, w_init, args=(x_ext, y_origin))
    W = result[0]
    plt.plot(x_predict, fit_function(W, make_x_ext(x_predict, order)))
    
plt.scatter(x_origin, y_origin)
plt.legend(['order=' + str(i) for i in np.arange(min_order, max_order+1, step)])
plt.show()
