import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_boston
from sklearn.preprocessing import add_dummy_feature

boston = load_boston()

X = boston.data[:, 5:6]
y = boston.target
print(X.shape, y.shape)  # (506, 1) (506,)

# 切片用に1の列を先頭に挿入する。
A = add_dummy_feature(X)
print(A)
# [[1.    6.575]
#  [1.    6.421]
#  [1.    7.185]
#  ...
#  [1.    6.976]
#  [1.    6.794]
#  [1.    6.03 ]]

# 最小二乗法を解く
coef, residuals, rank, singular = np.linalg.lstsq(A, y, rcond=None)
print(f"coefficient: {coef[1]}")
print(f"intercept: {coef[0]}")
# coefficient: 9.102108981180315
# intercept: -34.67062077643859


fig, ax = plt.subplots()

# 点を描画する。
ax.scatter(X.ravel(), y, s=10)

# 回帰曲線を描画する。
xs = np.linspace(*ax.get_xlim(), 100)
ys = coef[1] * xs + coef[0]
ax.plot(xs, ys, "r")

plt.show()
