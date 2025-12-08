import numpy as np

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


class PLS:
    def __init__(self, numComponentes):
        self.numComponentes = numComponentes
        
        # Médias e desvios para padronização
        self.x_media = None
        self.x_std = None
        self.y_media = None
        
        # Parâmetros aprendidos por componente
        self.phi_list = []        # Vetores de pesos para extrair cada componente
        self.theta_list = []      # Coeficientes que relacionam cada componente com y
        self.projcoef_list = []   # Coeficientes que representam como cada variável de X
                                  # contribui para o componente extraído

    # ---------------------------------------------------------
    # TREINAMENTO
    # ---------------------------------------------------------
    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)

        # ----------------------------
        # 1. Padronização de X e centralização de y
        # ----------------------------
        self.x_media = X.mean(axis=0)
        self.x_std = X.std(axis=0)
        Xp = (X - self.x_media) / self.x_std   # X padronizado

        self.y_media = y.mean()
        yp = y - self.y_media                  # y centralizado 

        n, d = X.shape
        y_pred_acum = np.zeros(n)

        # -----------------------------------------------------
        # 2. Loop para calcular cada componente do PLS
        # -----------------------------------------------------
        for m in range(self.numComponentes):

            # ---------------------------------------------------------
            # 2.1 Calcula o vetor de pesos para extrair o componente
            # ---------------------------------------------------------
            phi = Xp.T @ yp

            # Normalização para manter estabilidade
            phi = phi / np.linalg.norm(phi)

            self.phi_list.append(phi)

            # ---------------------------------------------------------
            # 2.2 Calcula o componente latente (combinação linear de X)
            # ---------------------------------------------------------
            z = Xp @ phi

            # ---------------------------------------------------------
            # 2.3 Relaciona o componente com y
            #     (coeficiente da regressão de y em z)
            # ---------------------------------------------------------
            theta = (z @ yp) / (z @ z)
            self.theta_list.append(theta)

            # Atualiza a predição acumulada
            y_pred_acum += theta * z

            # ---------------------------------------------------------
            # 2.4 Remove do X a parte alinhada com o componente z
            #
            #    Explicação:
            #    Depois que extraímos um componente (z), parte das
            #    informações de X já foi utilizada. Aqui removemos essa
            #    contribuição para permitir que o próximo componente
            #    capture uma nova direção independente.
            #
            #    coef = contribuição de cada coluna de X para z
            # ---------------------------------------------------------
            coef = (Xp.T @ z) / (z @ z)

            # Guardamos esse coeficiente para uso no predict
            self.projcoef_list.append(coef)

            # Remove a influência do componente atual de X
            Xp = Xp - np.outer(z, coef)

            print(f"Componente {m+1}: theta = {theta:.4f}")

        # Retorna a predição no conjunto de treino (opcional)
        return y_pred_acum + self.y_media

    # ---------------------------------------------------------
    # PREDIÇÃO EM NOVOS DADOS
    # ---------------------------------------------------------
    def predict(self, X):
        X = np.array(X)
        
        # Padronizamos X usando as estatísticas do treinamento
        Xp = (X - self.x_media) / self.x_std
        y_pred = np.zeros(X.shape[0])

        # Reproduzimos as extrações de componentes e as predições
        for phi, theta, coef in zip(self.phi_list, self.theta_list, self.projcoef_list):
            
            # 1. Extrai o componente do novo X
            z = Xp @ phi
            
            # 2. Usa esse componente para prever y
            y_pred += theta * z
            
            # 3. Remove da matriz X a parte influenciada por z
            #    (igual ao processo feito durante o treinamento)
            Xp = Xp - np.outer(z, coef)

        return y_pred + self.y_media

class RedeNeural:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        self.learning_rate = learning_rate
        
        # 1. Inicialização dos Pesos e Bias
        # Inicializamos com valores aleatórios pequenos
        np.random.seed(42)
        self.weights_input_hidden = np.random.randn(input_size, hidden_size)
        self.bias_hidden = np.zeros((1, hidden_size))
        
        self.weights_hidden_output = np.random.randn(hidden_size, output_size)
        self.bias_output = np.zeros((1, output_size))

    # Função de Ativação Sigmoide
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    # Derivada da Sigmoide
    def sigmoid_derivative(self, x):
        return x * (1 - x)

    # ---------------------------------------------------------
    # 2. Forward Phase (Propagação para frente)
    # ---------------------------------------------------------
    def forward(self, X):
        # Camada Oculta
        # v_j = soma(w * x) + b
        self.hidden_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        
        # y_j = phi(v_j)
        self.hidden_output = self.sigmoid(self.hidden_input)
        
        # Camada de Saída (linear para regressão)
        self.final_input = np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output
        self.final_output = self.final_input
        
        return self.final_output

    # ---------------------------------------------------------
    # 3. Backward Phase (Retropropagação)
    # ---------------------------------------------------------
    def backward(self, X, y, output):
        n_samples = X.shape[0]

        # --- Gradiente na camada de saída ---
        # Saída é linear → delta = erro direto
        delta_output = (output - y)

        # --- Gradiente da camada oculta ---
        # Propaga o erro para trás
        error_hidden = delta_output.dot(self.weights_hidden_output.T)

        # Multiplica pela derivada da sigmoide
        delta_hidden = error_hidden * self.sigmoid_derivative(self.hidden_output)

        # ------------------------------------------------
        # 4. Atualização dos Pesos (Regra Delta)
        # ------------------------------------------------

        # Hidden → Output
        self.weights_hidden_output -= self.learning_rate * self.hidden_output.T.dot(delta_output) / n_samples
        self.bias_output -= self.learning_rate * np.sum(delta_output, axis=0, keepdims=True) / n_samples

        # Input → Hidden
        self.weights_input_hidden -= self.learning_rate * X.T.dot(delta_hidden) / n_samples
        self.bias_hidden -= self.learning_rate * np.sum(delta_hidden, axis=0, keepdims=True) / n_samples

    # Treinamento completo
    def train(self, X, y, epochs):
        for i in range(epochs):
            # 1. Forward
            output = self.forward(X)
            
            # 2. Backward
            self.backward(X, y, output)
            
            # Reportar erro a cada 1000 épocas
            if i % 1000 == 0:
                loss = np.mean(np.square(y - output))
                print(f"Época {i}, Erro (MSE): {loss:.5f}")

    # Predição
    def predict(self, X):
        return self.forward(X)
