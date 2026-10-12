import numpy as np


def fit_ols(X, y):
    """最小二乘闭式解：w = (XᵀX)⁻¹Xᵀy

    参数
    ----
    X : np.ndarray, shape (n_samples, n_features)
        特征矩阵，不含截距列
    y : np.ndarray, shape (n_samples,)

    返回
    ----
    np.ndarray, shape (n_features + 1,)
        w[0] 是截距项，w[1:] 是各特征系数
    """
    # 第 1 步：给 X 左边拼一列 1（把截距吸收进 w）
    ones = np.ones((X.shape[0], 1))     # n 行 1 列的全 1
    X_aug = np.hstack([ones, X])        # shape: (n, m+1)

    # 第 2 步：算 XᵀX 和 Xᵀy（正规方程的两个"原材料"）
    XtX = X_aug.T @ X_aug               # shape: (m+1, m+1)
    Xty = X_aug.T @ y                   # shape: (m+1,)

    # 第 3 步：解线性方程组 XᵀX·w = Xᵀy，而不是算 (XᵀX)⁻¹ 再乘
    w = np.linalg.solve(XtX, Xty)

    return w

def fit_gd(X, y, lr=0.01, epochs=1000):
    """梯度下降拟合线性回归：从随机点出发，沿下坡方向反复挪步。

    参数
    ----
    X : np.ndarray, shape (n_samples, n_features)
        特征矩阵，不含截距列
    y : np.ndarray, shape (n_samples,)
    lr : float
        学习率（步长），越大走得越快、但容易过冲震荡
    epochs : int
        迭代次数（最多挪多少步）

    返回
    ----
    np.ndarray, shape (n_features + 1,)
        w[0] 是截距项，w[1:] 是各特征系数
    """
    # 和 fit_ols 一样：拼一列 1，把截距吸收进 w
    ones = np.ones((X.shape[0], 1))
    X_aug = np.hstack([ones, X])
    n = X.shape[0]

    # 1. 随便猜一个起点：全 0（也可以随机）
    w = np.zeros(X_aug.shape[1])

    # 2. 反复挪步
    for _ in range(epochs):
        error = X_aug @ w - y              # 当前预测 - 真实值（每个样本偏了多少）
        gradient = 2 / n * (X_aug.T @ error)   # 坡度：往哪边下山最快
        w = w - lr * gradient              # 沿下坡挪一小步

    return w
