import numpy as np
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
    """
      - 1 camada oculta
      - sigmoide
      - regularização L2 (weight decay)
      - treino por gradiente descendente
    """

    def __init__(self, input_size, hidden_size, output_size=1,
                 learning_rate=0.01, weight_decay=0.0):
        
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay   # L2 regularization

        # Pesos pequenos, como recomendado no livro
        limit_in = 1 / np.sqrt(input_size)
        limit_hidden = 1 / np.sqrt(hidden_size)

        np.random.seed(42)

        self.W1 = np.random.uniform(-limit_in, limit_in, (input_size, hidden_size))
        self.b1 = np.zeros((1, hidden_size))

        self.W2 = np.random.uniform(-limit_hidden, limit_hidden, (hidden_size, output_size))
        self.b2 = np.zeros((1, output_size))


    # --- FUNÇÕES DE ATIVAÇÃO ---
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_deriv(self, x):
        return x * (1 - x)


    # --- FORWARD ---
    def forward(self, X):
        self.Z1 = np.dot(X, self.W1) + self.b1
        self.A1 = self.sigmoid(self.Z1)

        self.Z2 = np.dot(self.A1, self.W2) + self.b2   # saída linear
        self.A2 = self.Z2

        return self.A2


    # --- BACKPROP COM DECAY ---
    def backward(self, X, y, output):
        n = len(X)

        # erro da saída
        dZ2 = (output - y) / n

        # gradiente com regularização L2
        dW2 = np.dot(self.A1.T, dZ2) + self.weight_decay * self.W2
        db2 = np.sum(dZ2, axis=0, keepdims=True)

        # erro da camada oculta
        dA1 = np.dot(dZ2, self.W2.T)
        dZ1 = dA1 * self.sigmoid_deriv(self.A1)

        # gradiente com L2
        dW1 = np.dot(X.T, dZ1) + self.weight_decay * self.W1
        db1 = np.sum(dZ1, axis=0, keepdims=True)

        # atualização
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1


    # --- TREINAMENTO ---
    def train(self, X, y, epochs=5000, verbose=True):
        for ep in range(epochs):
            out = self.forward(X)
            self.backward(X, y, out)

            if verbose and ep % 1000 == 0:
                mse = np.mean((y - out) ** 2)
                print(f"Época {ep}, MSE = {mse:.5f}")


    def predict(self, X):
        return self.forward(X)
