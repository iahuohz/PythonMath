import numpy as np

# (1)reduce：
# 以np.add.identity作为初始参数(I)，将参数数组中第二个元素与I相加，结果作为新的I
# 依次类推，最后返回I
a = np.arange(1, 10)
b = a.reshape((3,3))
print("b", b)
print("Reduce add a", np.add.reduce(a))
print("Reduce add b", np.add.reduce(b, axis=1))
print("Reduce multiply a", np.multiply.reduce(a))

# (2)accumulate
# 类似于cumadd, cumprod等，将依次为每个元素调用reduce
a = np.arange(1, 10)
print(np.add.accumulate(a))

# (3) outter
a = np.arange(1, 10)
print("Outer", np.add.outer([2,3,7], a))