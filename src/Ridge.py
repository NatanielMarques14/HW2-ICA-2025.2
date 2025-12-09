import numpy as np
import numpy.linalg as npla

def ridge_train_beta(X, y, lambda_value):
    
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1,)
    y_mean = y.mean()
    yc = y - y_mean
    XtX = X.T @ X
    p = XtX.shape[0]
    A = XtX + lambda_value * np.eye(p)
    Xty = X.T @ yc
    beta = npla.solve(A, Xty)   # mais estável que inv(...)
    intercept = y_mean
    return beta, intercept

def ridge_predict(beta, intercept, X):
    X = np.asarray(X, dtype=float)
    return X @ beta + intercept

def ridge_factory(lam):
    return lambda Xtr, ytr, Xval: ridge_predict(*ridge_train_beta(Xtr, ytr, lam), Xval)

def Best_ridge_lambda(ridge_means, lambdas):
    mask_ridge_valido = ridge_means >= 0                 # Considera apenas RMSE válidos
    rmse_ridge_validos = ridge_means[mask_ridge_valido]
    indices_ridge_validos = np.arange(len(ridge_means))[mask_ridge_valido]
    best_ridge_idx = indices_ridge_validos[np.argmin(rmse_ridge_validos)]
    return lambdas[best_ridge_idx], rmse_ridge_validos[best_ridge_idx]