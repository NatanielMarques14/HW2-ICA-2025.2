import numpy as np
from classes_base import PLS, RedeNeural
import time

# aqui basicamente vamos só implementar a fórmula do rmse mostrada no notebook
def rmse(y_real, y_pred):
    y_real = np.array(y_real)
    y_pred = np.array(y_pred)
    return np.sqrt(np.mean((y_real - y_pred)**2))

# o mesmo feito com o r2
def r2(y_real, y_pred):
    y_real = np.array(y_real)
    y_pred = np.array(y_pred)
    ss_res = np.sum((y_real - y_pred)**2)
    ss_tot = np.sum((y_real - np.mean(y_real))**2)
    return 1 - ss_res/ss_tot

def cross_validate(model_fn, X, y, k=5):
    # esse model_fn é uma função de um modelo há ser usado, que recebe X_train_fold, y_train_fold, X_val_fold e retorna y_pred)
    X = np.array(X)
    y = np.array(y)
    fold_tam = len(X) // k
    rmses = []
    r2s = []

    # loop explicado no notebook
    for i in range(k):
        ini = i * fold_tam
        fim = ini + fold_tam

        X_val = X[ini:fim]
        y_val = y[ini:fim]

        X_train = np.concatenate([X[:ini], X[fim:]])
        y_train = np.concatenate([y[:ini], y[fim:]])

        y_pred = model_fn(X_train, y_train, X_val)

        rmses.append(rmse(y_val, y_pred))
        r2s.append(r2(y_val, y_pred))

    return np.mean(rmses), np.mean(r2s)

def fit_ols(X,y):
    # aqui vamos treinar o OLS com a forma matricial
    X = np.array(X)
    y = np.array(y)
    
    # adicionamos o termo de bias, intercepto beta zero, uma coluna de 1s, para podermos realizar a multiplicação matricial com esse X aumentado:
    X_aum = np.column_stack([np.ones(len(X)), X])
    # isso garante que o primeiro coeficiente beta zero seja multiplicado por 1 em todas as linhas
    
    # beta = (X^T X)^-1 X^T y
    beta = np.linalg.inv(X_aum.T @ X_aum) @ (X_aum.T @ y)
    return beta

def predict_ols(X, beta):
    X = np.array(X)
    X_aum = np.column_stack([np.ones(len(X)), X])
    return X_aum @ beta

def pls_model(n_components):
    def model_fn(X_train, y_train, X_val):
        model = PLS(numComponentes=n_components)
        model.fit(X_train, y_train)
        return model.predict(X_val)
    return model_fn

def evaluate_lambda_grid(factory_fn, X, y, lambdas, k=5, verbose=True):
    means, stds, times = [], [], []
    for lam in lambdas:
        model_fn = factory_fn(lam)
        t0 = time.time()
        rmse_mean, _ = cross_validate(model_fn, X, y, k=k)
        t1 = time.time()
        means.append(rmse_mean)
        stds.append(0)
        times.append(t1 - t0)
        if verbose:
            print(f'λ={lam:.2e} → RMSE mean={means[-1]:.4f} time={times[-1]:.1f}s')
    return np.array(means), np.array(stds), np.array(times)
