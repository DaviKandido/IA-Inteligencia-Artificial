# Importação de bibliotecas
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from enum import Enum

"""
Algoritimo em Alto Nivel

ArvoreDecesion(dataSet=dataSet, classe='Conclusao', criterio='entropia'):
 1) Verifica se hora de parada (se todos os exemplos da classe forem iguais, não há dados para classificação)
 2) Escolhe o "melhor" atributo
  2.1) Calcula a entropia da classe
  2.2) Calcula a entropia dos atributos
  2.3) Seleciona o atributo com maior ganho de informação
 3) Cria uma aresta por valor do atributo
 4) Cria um nó (filho) por valor
  4.1) Separa o subconjunto para sub valor do atributo filho
 5) Chama a função recursivamente
"""

# ----------------------------
# ENUMS PARA CRITÉRIO E MÉTODO
# ----------------------------
class Algoritmos(Enum):
    ID3 = "ID3"
    C45 = "C4.5"
    CART = "CART"


# ----------------------------
# CLASSE NO
# ----------------------------
class No:
    def __init__(self, atributo=None, valor=None, sample = 0, grauConfusao=None, value=None, filhos=None, classe=None):
        """
        Inicializa um No com os parâmetros dados.

        Parâmetros:
        atributo (str): Atributo a ser testado no No intermédio.
        sample (int): Quantidade de Exemplos por no.
        grauConfusao (float): Valor da Entropia ou Geni.
        value (list): Quantidade de Exemplos em cada classe.
        filhos (list): Usado quando for nó intermédio.
        classe (str): Usado quando for nó folha (Classe final).
        """
        
        self.atributo = atributo                                  # usado quando for nó intermédio (Atributo a ser testado)
        self.valor = valor                                        # usado quando for nó intermédio
        self.grauConfusao = grauConfusao                          # Valor da Entropia ou Geni 
        self.value = list(value) if value is not None else []     # Quantidade de Exemplos em cada classe
        self.sample = sample                                      # Quantidade de Exemplos por no
        self.filhos = list(filhos) if filhos is not None else []  # usado quando for nó intermédio
        self.classe = classe                                      # usado quando for nó folha (Classe final)

    def eh_folha(self):
        return self.classe is not None

    def eh_intermediario(self):
        return not self.eh_folha()


# ----------------------------
# CLASSES DA ÁRVORE
# ----------------------------
class Arvore:
  def __init__(self, raiz=None, regras=None):
        """
        Inicializa uma arvore com o No raiz e as regras definidas.

        Parâmetros:
        raiz (No): O No raiz da arvore.
        regras (dict): Um dicionário com as regras.
        """
        self.raiz = raiz
        self.regras = regras

  def imprimir(self, no=None, nivel=0):
    no = no or self.raiz
    prefixo = "  " * nivel

    if no.eh_folha():
        print(f"{prefixo}- Classe: {no.classe} (amostra: {no.sample})")
    else:
        print(f"{prefixo}- Atributo: {no.atributo} (impureza: {no.grauConfusao:.4f}, amostra: {no.sample})")
        for filho in no.filhos:
            print(f"{prefixo}  Valor da aresta: {filho.valor}")
            # Chama recursivamente passando o próprio nó filho
            self.imprimir(filho, nivel + 2)




# ----------------------------
# FUNÇÕES DE IMPUREZA - ENTROPIA E GINI
# ----------------------------
def entropia(labels):
  """
    Calcula a entropia a partir de uma série de rótulos (labels).
  """
  valores, contagens = np.unique(labels, return_counts=True)
  probs = contagens / contagens.sum()
  return -np.sum(probs * np.log2(probs))

def gini(labels):
    """
    Calcula o índice de Gini a partir de uma série de rótulos (labels).
    """
    valores, contagens = np.unique(labels, return_counts=True)
    probs = contagens / contagens.sum()
    return 1 - np.sum(probs**2)


# ----------------------------
# CONDIÇÃO DE PARADA
# ----------------------------
def hora_de_parada(dataSet: pd.DataFrame, classe: str='Conclusao'):
  """
    Verifica se a hora de parada foi atingida.

    Se todos os exemplos da classe forem iguais, ou se não houver dados para classificação,
    a hora de parada foi atingida.

    Parâmetros:
    dataSet (pd.DataFrame): Conjunto de dados.
    classe (str): Nome da coluna que contém a classe.

    Retorna:
    bool: Verdadeiro se a hora de parada foi atingida, falso caso contrário.
  """
  return len(dataSet[classe].unique()) == 1 or len(dataSet) == 0


# ----------------------------
# ESCOLHA DO MELHOR ATRIBUTO
# ----------------------------
def escolher_melhor_atributo(dataSet: pd.DataFrame, classe: str, algoritimo: Algoritmos):
    """
    Escolhe o melhor atributo para construir a árvore.

    Parâmetros:
    dataSet (pd.DataFrame): Conjunto de dados.
    classe (str): Nome da coluna que contém a classe.
    algoritmo (Algoritmos): Algoritmo a ser utilizado (ID3, C4.5 ou CART).

    Retorna:
    str: O nome do melhor atributo.
    """
    atributos = [col for col in dataSet.columns if col != classe]
    if algoritimo == algoritimo.CART:
        base_score = gini(dataSet[classe])
    else:
        base_score = entropia(dataSet[classe])

    melhor_atributo = None
    melhor_ganho = -np.inf

    for atributo in atributos:
        score_atributo = 0
        split_info = 0
        for valor in dataSet[atributo].unique():
            subset = dataSet[dataSet[atributo] == valor]
            peso = len(subset) / len(dataSet)
            if algoritimo == algoritimo.CART:
                score_atributo += peso * gini(subset[classe])
            else:
                score_atributo += peso * entropia(subset[classe])
                split_info -= peso * np.log2(peso) if peso > 0 else 0

        ganho = base_score - score_atributo
        if  algoritimo == Algoritmos.C45 and split_info > 0:
            ganho = ganho / split_info

        if ganho > melhor_ganho:
            melhor_ganho = ganho
            melhor_atributo = atributo

    return melhor_atributo


# ----------------------------
# CONSTRUÇÃO DA ÁRVORE
# ---------------------------
def ArvoreDecesion(dataSet: pd.DataFrame, classe: str='Conclusao', algoritimo: Algoritmos = Algoritmos.ID3):
  """
  Constrói uma árvore de decisão a partir do conjunto de dados fornecido.

  Parâmetros:
  dataSet (pd.DataFrame): Conjunto de dados.
  classe (str): Nome da coluna que contém a classe. (Padrão: 'Conclusao')
  algoritmo (Algoritmos): Algoritmo a ser utilizado (ID3, C4.5 ou CART). (Padrão: Algoritmos.ID3)

  Retorna:
  Arvore: A árvore de decisão construída.
  """
  arvore = Arvore()
  
  arvore.raiz = No()
  arvore.raiz = construir_arvore(dataSet=dataSet, no=arvore.raiz, classe=classe, algoritimo=algoritimo)
  return arvore


def construir_arvore(dataSet: pd.DataFrame, no: No, classe: str, algoritimo: Algoritmos):
  
  # Preenche estatísticas do nó atual
  no.sample = len(dataSet)
  no.value = dataSet[classe].value_counts().tolist()

  if algoritimo == algoritimo.CART:
    no.grauConfusao = gini(dataSet[classe])
  else:
    no.grauConfusao = entropia(dataSet[classe])
  
  # condição de parada
  if hora_de_parada(dataSet, classe) or len(dataSet.columns) == 1:
    no.classe = dataSet[classe].value_counts().idxmax()
    return no

  # escolha do melhor atributo
  melhor_atributo = escolher_melhor_atributo(dataSet, classe, algoritimo)

  no.atributo = melhor_atributo

  # cria filhos
  for valor in dataSet[melhor_atributo].unique():
    subset = dataSet[dataSet[melhor_atributo] == valor]
    # cria o filho com a informação da aresta 
    # chama recursivamente para construir o nó do subset
    filho = construir_arvore(subset, No(valor = valor), classe, algoritimo)
    # adiciona o filho na lista
    no.filhos.append(filho)

  
  print(no.value)
  return no




# ----------------------------
# EXEMPLO DE USO
# ----------------------------
if __name__ == "__main__":
    
    # Carrega dados
    data = pd.read_csv('./DataBases/restaurante/restaurante.csv', sep=';')

    # Construir árvore ID3
    arvore_id3 = ArvoreDecesion(data, classe='Conclusao', algoritimo=Algoritmos.ID3)
    # Construir árvore C4.5
    arvore_c45 = ArvoreDecesion(data, classe='Conclusao', algoritimo=Algoritmos.C45)
    # Construir árvore CART
    arvore_cart = ArvoreDecesion(data, classe='Conclusao', algoritimo=Algoritmos.CART)

    # Imprime árvore com ID3
    print(" \n------------ Árvore ID3: ------------\n")
    arvore_id3.imprimir()

    # Imprime árvore com ID3
    print("\n------------ Árvore C4 .5: ------------\n")
    arvore_c45.imprimir()

    # Imprime árvore com ID3
    print(" \n------------ Árvore CART: ------------\n")
    arvore_cart.imprimir()
