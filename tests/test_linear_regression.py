import numpy as np
from market_pipeline.linear_regression import fit_ols, fit_gd


def test_fit_ols_recovers_true_coefficients():
    rng = np.random.default_rng(42)
    n = 200
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    X = np.column_stack([x1, x2])
    # 真实规律：y = 2 + 3·x1 - 1·x2 + 小噪声
    y = 2 + 3 * x1 - 1 * x2 + rng.normal(scale=0.1, size=n)

    w = fit_ols(X, y)

    assert abs(w[0] - 2) < 0.1   # 截距 ≈ 2
    assert abs(w[1] - 3) < 0.1   # x1 系数 ≈ 3
    assert abs(w[2] + 1) < 0.1   # x2 系数 ≈ -1

def test_fit_gd_converges_to_ols():
    rng = np.random.default_rng(42)
    n = 200
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    X = np.column_stack([x1, x2])
    y = 2 + 3 * x1 - 1 * x2 + rng.normal(scale=0.1, size=n)

    w_gd = fit_gd(X, y, lr=0.1, epochs=2000)
    w_ols = fit_ols(X, y)

    # 两条完全不同的路，应该走到同一个答案
    np.testing.assert_allclose(w_gd, w_ols, atol=0.05)
