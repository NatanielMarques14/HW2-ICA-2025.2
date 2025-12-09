import numpy as np
def soft_threshold_scalar(rho, lam):
    if rho < -lam:
        return rho + lam
    elif rho > lam:
        return rho - lam
    else:
        return 0.0

def lasso_cd_train(X, y, lambda_value, tol=1e-6, max_iter=1000, verbose=False):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1,)
    y_mean = y.mean()
    yc = y - y_mean
    Xc = X - X.mean(axis=0)
    col_norms = np.sqrt((Xc**2).sum(axis=0))
    col_norms[col_norms == 0] = 1.0
    Xs = Xc / col_norms
    n, p = Xs.shape
    beta = np.zeros(p)
    for it in range(max_iter):
        beta_old = beta.copy()
        for j in range(p):
            r_j = yc - Xs @ beta + Xs[:, j] * beta[j]
            rho = (Xs[:, j] * r_j).sum()
            beta_j = soft_threshold_scalar(rho, lambda_value/2) / (Xs[:, j]**2).sum()
            beta[j] = beta_j
        maxdiff = np.max(np.abs(beta - beta_old))
        if verbose:
            print(f'iter {it} maxdiff {maxdiff:.3e}')
        if maxdiff < tol:
            break
    beta_orig = beta / col_norms
    intercept = y_mean - (X.mean(axis=0) @ beta_orig)
    return beta_orig, intercept

def lasso_predict(beta, intercept, X):
    X = np.asarray(X, dtype=float)
    return X @ beta + intercept

def lasso_factory(lam):
    return lambda Xtr, ytr, Xval: lasso_predict(*lasso_cd_train(Xtr, ytr, lam), Xval)

def Best_lasso_lambda(lasso_means, lambdas):
    mask_lasso_valido = lasso_means >= 0                 # Considera apenas RMSE válidos
    rmse_lasso_validos = lasso_means[mask_lasso_valido]
    indices_lasso_validos = np.arange(len(lasso_means))[mask_lasso_valido]
    best_lasso_idx = indices_lasso_validos[np.argmin(rmse_lasso_validos)]
    return lambdas[best_lasso_idx], rmse_lasso_validos[best_lasso_idx]