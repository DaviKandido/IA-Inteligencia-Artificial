# %%
import numpy as np
import matplotlib.pyplot as plt
import pickle

# ### Etapas:
# 
# 1. Inicialização dos pesos e bias
# 2. FeedForward
# 3. Calculo da perda
# 4. Backpropagation
# 5. Fit

class RnModel:

  def __init__(self, x: np.ndarray, y: np.ndarray, hidden_neurons: int = 10, 
                output_neurons: int = 1, random_seed: int | None = None, activation: str = 'sigmoid'):
    """
      Inicializa uma rede neural simples com uma camada oculta.

      Parâmetros:
      ----------
      x : np.ndarray
          Matriz de entrada com formato (n_amostras, n_features).
      y : np.ndarray
          Vetor ou matriz de rótulos com formato (n_amostras,) ou (n_amostras, n_outputs).
      hidden_neurons : int
          Número de neurônios na camada oculta.
      output_neurons : int
          Número de neurônios na camada de saída.
      random_seed : int | None
          Valor para reprodutibilidade da inicialização aleatória dos pesos.
      activation : str
          Nome da função de ativação usada na camada oculta. 
          Opções: 'tanh', 'relu', 'sigmoid'.

      Descrição:
      ----------
      Este construtor configura a estrutura básica da rede:
        - Valida as dimensões de entrada.
        - Define os tamanhos das camadas.
        - Inicializa pesos e bias com pequenas variações aleatórias (método Xavier).
        - Define a função de ativação escolhida em activation.
        - Armazena parâmetros principais e histórico de perda.
    """

    # Validações simples
    if x.ndim != 2:
      raise ValueError("x deve ser uma matriz 2D com shape (n_amostras, n_features)")
    if y.ndim not in (1, 2):
      raise ValueError("y deve ser um vetor 1D ou matriz 2D de rótulos")

    self.x = x
    # garantir formato coluna para y quando saída única
    self.y = y.reshape(-1, 1) if (y.ndim == 1 and output_neurons == 1) else y

    self.hidden_neurons = int(hidden_neurons)
    self.output_neurons = int(output_neurons)
    
    # dimensões que é o Nº de entradas é o numero de entradas (colunas na entrada x)
    self.input_neurons = x.shape[1]

    # inicialização de pesos (pequenos valores aleatórios) e biases (zeros)
    # Xavier Inicialization -> Variancia dos pesos igual em todas as camadas
    rng = np.random.RandomState(random_seed)
    
    # Camada de entrada para a camada oculta
    self.w1 = rng.randn(self.input_neurons, self.hidden_neurons) / np.sqrt(self.input_neurons) # (n_features, hidden)
    self.b1 = np.zeros((1, self.hidden_neurons)) # (1, hidden)
    
    # Camada oculta para a camada de saída                     
    self.w2 = rng.randn(self.hidden_neurons, self.output_neurons) / np.sqrt(self.hidden_neurons) # (hidden, output)
    self.b2 = np.zeros((1, self.output_neurons)) # (1, output)
    self.z1 = 0
    self.f1 = 0
    self.activation = activation.lower()

    self.model_dist = {
      'w1': self.w1,
      'b1': self.b1,
      'w2': self.w2,
      'b2': self.b2,
      'random_seed': random_seed,
      'activation': self.activation
    }

    # histórico de perda
    self.loss_history = []

  # -------------------- Funções auxiliares --------------------

  def showPlot(self, predictions):
    """
      Plota um scatter 2D das amostras coloridas pelas predições.

      Parâmetros:
      ----------
      predictions : np.ndarray
          Vetor de rótulos ou previsões com comprimento igual ao número de amostras (n_amostras,).

      Retorna:
      --------
      None

      Descrição:
      ----------
      Exibe um gráfico de dispersão usando as duas primeiras características de entrada
      (x[:, 0] e x[:, 1]) e colore cada ponto de acordo com 'predictions'. A função
      exige que os dados de entrada tenham exatamente 2 features.
    """
    if self.x.shape[1] != 2:
      raise ValueError("showPlot só funciona para entradas 2D com shape (n_amostras, 2)")

    plt.scatter(self.x[:, 0], self.x[:, 1], s=50, c=predictions, cmap='rainbow', alpha=0.7)
    plt.xlabel('Feature 0')
    plt.ylabel('Feature 1')
    plt.title('Predições (showPlot)')
    plt.show()


  def activation_forward(self, z):
    """Aplica a função de ativação selecionada."""
    if self.activation == "tanh":
      return np.tanh(z)
    elif self.activation == "relu":
      return np.maximum(0, z)
    elif self.activation == "sigmoid":
      return 1 / (1 + np.exp(-z))
    else:
      raise ValueError(f"Função de ativação '{self.activation}' não suportada.")
      
  # ------------------------------------------------------------

  def activation_derivative(self, z):
    """Calcula a derivada da função de ativação selecionada."""
    if self.activation == "tanh":
      return (1 - np.power(np.tanh(z), 2))
    elif self.activation == "relu":
      return (z > 0).astype(float)
    elif self.activation == "sigmoid":
      s = 1 / (1 + np.exp(-z))
      return s * (1 - s)
    else:
      raise ValueError(f"Função de ativação '{self.activation}' não suportada.")

  def forward(self, x: np.ndarray) -> np.ndarray:
    """
      Realiza a etapa de propagação direta (forward pass).

      Parâmetros:
      ----------
      x : np.ndarray
          Entradas da rede com formato (n_amostras, n_features).

      Retorna:
      --------
      np.ndarray
          Saída da rede após a função activations escolhida .

      Descrição:
      ----------
      Executa o fluxo de dados da entrada até a saída:
        1. Calcula z1 = x·w1 + b1.
        2. Aplica a função de ativação tanh/sigmoid/relu: f1 = f(z1).
        3. Calcula z2 = f1·w2 + b2.
        4. Aplica a função de ativação final (Sigmoid ou Softmax).
    """
    # Equação da reta (1)
    # Multiplicação de matrizes + bias 
    self.z1 = x.dot(self.w1) + self.b1

    # Função de ativação (1)
    self.f1 = self.activation_forward(self.z1)

    # Equação da reta (2)
    z2 = self.f1.dot(self.w2) + self.b2
    
    # Se for problema binário (1 neurônio na saída), use sigmoid
    if self.output_neurons == 1:
        return 1 / (1 + np.exp(-z2)) # Sigmoid na saída
    else:
    # Softmax (Probabilidade de cada classe) -> Função de ativação (2)
        exp_values = np.exp(z2 - np.max(z2, axis=1, keepdims=True)) # Subtrai o max para estabilidade numérica
        softmax = exp_values / np.sum(exp_values, axis = 1, keepdims = True)
        return softmax


  def loss(self, output):
    """
      Calcula a função de perda (erro) utilizando entropia cruzada.

      Parâmetros:
      ----------
      output : np.ndarray
          Saídas previstas da rede (probabilidades de cada classe).

      Retorna:
      --------
      float
          Valor médio da perda para a época atual.

      Descrição:
      ----------
      Mede o erro entre as previsões do modelo e os rótulos corretos.
    """
    if self.output_neurons == 1:
      # Binary Cross-Entropy
      # L = -y * log(y_hat) - (1-y) * log(1-y_hat)
      eps = 1e-9
      # Garantir que output e self.y tenham o mesmo shape (N, 1)
      output = output.reshape(-1, 1)
      y = self.y.reshape(-1, 1)
      return -np.mean(y * np.log(output + eps) + (1 - y) * np.log(1 - output + eps))
    else:
      # Categorical Cross-Entropy
      # y.ravel() deve conter os índices de classe (0, 1, 2...)
      y_indices = self.y.ravel().astype(int)
      # Pega a probabilidade predita para a classe correta
      correct_probabilities = output[range(self.x.shape[0]), y_indices]
      log_prob = -np.log(correct_probabilities + 1e-9)
      return np.mean(log_prob)


  def backpropagation(self, outputs: np.ndarray, learning_rate: float = 0.1):
    """
      Realiza o processo de retropropagação do erro (backpropagation).

      Parâmetros:
      ----------
      outputs : np.ndarray
          Saídas obtidas da etapa de forward.
      learning_rate : float
          Taxa de aprendizado usada para ajustar os pesos.

      Descrição:
      ----------
      Calcula os gradientes do erro em relação aos pesos e bias,
      e ajusta os parâmetros do modelo com base no gradiente descendente:
        - Calcula os deltas das camadas (delta2 e delta1).
        - Obtém gradientes dw e db.
        - Atualiza pesos e bias com o fator de aprendizado definido.
    """
    
    if self.output_neurons == 1:
      # CASO BINÁRIO (Sigmoid na saída, Binary Cross-Entropy)
      # Delta 2 = Saída - Alvo (simples para BCE + Sigmoid)
      delta2 = outputs - self.y # <-- Correção APLICADA
        
    else:
      # CASO MULTI-CLASSE (Softmax na saída, Categorical Cross-Entropy)
      delta2 = np.copy(outputs)
      # self.y é a matriz de índices de classe
      y_indices = self.y.ravel().astype(int)
      # Diminui 1 (o valor do rótulo real) apenas na posição da classe correta
      delta2[range(self.x.shape[0]), y_indices] -= 1

    # Gradientes da camada de saída
    dw2 = (self.f1.T).dot(delta2)
    db2 = np.sum(delta2, axis = 0, keepdims=True)

    # Gradientes da camada oculta (derivada da tanh/sigmoid/relu)
    delta1 = delta2.dot(self.w2.T) * self.activation_derivative(self.z1)
    dw1 = (self.x.T).dot(delta1)
    db1 = np.sum(delta1, axis = 0, keepdims=True)

    # Atualização dos pesos e bias
    self.w1 += -learning_rate * dw1 # 'w1 = w1 - self.learning_rate*dw1'
    self.w2 += -learning_rate * dw2 
    self.b1 += -learning_rate * db1 
    self.b2 += -learning_rate * db2 
    


  def fit(self, learning_rate: float = 0.1, epochs: int = 1000):
    """
      Treina o modelo usando o algoritmo de descida do gradiente (Gradient Descent).

      Parâmetros:
      ----------
      learning_rate : float
          Taxa de aprendizado que define o tamanho do passo na atualização dos pesos.
      epochs : int
          Número de iterações (épocas) de treinamento.

      Retorna:
      --------
      np.ndarray
          Predições finais do modelo após o término do treinamento.

      Descrição:
      ----------
      Para cada época:
        1. Executa a propagação direta (forward).
        2. Calcula a perda média (erro).
        3. Atualiza pesos e bias pela retropropagação.
        4. Mede a acurácia com base nas predições atuais.
      Exibe periodicamente métricas de desempenho.
    """
    for epoch in range(epochs):
      outputs = self.forward(self.x)
      loss = self.loss(outputs)
      self.loss_history.append(loss)
      self.backpropagation(outputs, learning_rate)

      # Calculo de acuracia
      y_true = self.y.ravel()
      if self.output_neurons == 1:
        prediction = (outputs > 0.5).astype(int).ravel()
      else:
        # Acurácia para multi-classe (argmax)
        prediction = np.argmax(outputs, axis=1)
        # O y_true precisa ser o índice da classe (não one-hot)
        y_true = y_true.astype(int) # <-- Correção APLICADA
        
      correct = np.sum(prediction == y_true)
      accuracy = correct / len(y_true)
      

      # Correção na condição de exibição de progresso
      if (epoch + 1) % max(1, (epochs // 10)) == 0:
        print(f'Epoch: [{epoch+1} / {epochs}] Accuracy: {accuracy:.3f} Loss: {loss:.4f} Correct: {correct} Total: {self.y.shape[0]}')
    
    return prediction

    # -------------------- Save and Load --------------------

  def save(self, filename):
    with open(filename, 'wb') as f:
        pickle.dump(self.model_dist, f)

  def load(self, filename):
    with open(filename, 'rb') as f:
        self.model_dist = pickle.load(f)
    self.w1 = self.model_dist['w1']
    self.b1 = self.model_dist['b1']
    self.w2 = self.model_dist['w2']
    self.b2 = self.model_dist['b2']
    self.random_seed = self.model_dist['random_seed']
    self.activation = self.model_dist['activation']


