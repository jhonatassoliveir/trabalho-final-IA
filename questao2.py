import os
import sys
import numpy as np
import pandas as pd
import torch
import ltn
from itertools import product

# Define o dispositivo para treinamento: usa GPU se disponível, senão CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BoardClassifier(torch.nn.Module):
    """
    Modelo de rede neural simples para classificar se um tabuleiro de Sudoku é solucionável.
    A entrada é um tabuleiro flatten (dimensão [batch, size*size]), a saída é um valor entre 0 e 1.
    """
    def __init__(self, size):
        super().__init__()
        self.linear1 = torch.nn.Linear(size * size, 128)
        self.linear2 = torch.nn.Linear(128, 64)
        self.linear3 = torch.nn.Linear(64, 1)

    def forward(self, x):
        x = x.view(-1, x.shape[1] * x.shape[2])  # Achata o tabuleiro para [batch, features]
        x = torch.relu(self.linear1(x))
        x = torch.relu(self.linear2(x))
        return torch.sigmoid(self.linear3(x))


def criar_dados(board_size, n_amostras):
    """
    Gera dois conjuntos de dados:
    - Tabuleiros parcialmente preenchidos mas resolvíveis
    - Tabuleiros com erros de repetição (não resolvíveis)
    
    Args:
        board_size (int): Tamanho do tabuleiro (ex: 9 para 9x9).
        n_amostras (int): Quantidade de exemplos por tipo.
    
    Returns:
        Tuple[Tensor, Tensor]: (resolvíveis, não resolvíveis)
    """
    resolviveis, falhos = [], []

    for _ in range(n_amostras):
        # Tabuleiro válido com permutação simples das linhas
        base = np.random.permutation(board_size) + 1
        base_board = np.zeros((board_size, board_size), dtype=int)
        for i in range(board_size):
            base_board[i] = np.roll(base, i)

        # Aplica uma máscara binária para esconder valores
        mask = np.random.binomial(1, 0.5, size=(board_size, board_size))
        b_valido = base_board * mask
        resolviveis.append(b_valido)

        # Cria erro proposital (duplicação na mesma linha)
        b_invalid = base_board.copy()
        for _ in range(2):
            row = np.random.randint(0, board_size)
            col1, col2 = np.random.choice(board_size, 2, replace=False)
            b_invalid[row, col2] = b_invalid[row, col1]
        falhos.append(b_invalid)

    return (
        torch.tensor(np.array(resolviveis), dtype=torch.float32),
        torch.tensor(np.array(falhos), dtype=torch.float32)
    )


def treinar_modelo(modelo_pred, size):
    """
    Treina um predicado LTN para identificar tabuleiros solucionáveis e não solucionáveis.
    
    Args:
        modelo_pred (ltn.Predicate): Predicado com modelo embutido.
        size (int): Tamanho dos tabuleiros.
    """
    treino_bom, treino_ruim = criar_dados(size, 150)
    treino_bom, treino_ruim = treino_bom.to(device), treino_ruim.to(device)

    quant = ltn.Quantifier(ltn.fuzzy_ops.AggregPMeanError(), quantifier="f")
    variavel_b = ltn.Variable("bom", treino_bom)
    variavel_r = ltn.Variable("ruim", treino_ruim)

    otimizador = torch.optim.Adam(modelo_pred.model.parameters(), lr=0.001)

    for ep in range(700):
        otimizador.zero_grad()
        sat1 = quant(variavel_b, modelo_pred(variavel_b))  # Deve ser verdadeiro
        sat2 = quant(variavel_r, modelo_pred(variavel_r))  # Deve ser falso
        perda = 1. - sat1.value + sat2.value

        perda.backward()
        otimizador.step()
        if ep % 200 == 0:
            print(f"Época {ep} | Satisfação: {sat1.value.item():.3f} / {sat2.value.item():.3f}")
    print("Treinamento finalizado.")


def tem_num_bloqueado(tabuleiro):
    """
    Verifica se existe algum número de 1 a N que não pode ser colocado em nenhuma célula vazia.
    
    Args:
        tabuleiro (ndarray): Matriz numpy representando o tabuleiro.
    
    Returns:
        Tuple[bool, int]: (bloqueado, número_bloqueado)
    """
    n = tabuleiro.shape[0]
    raiz = int(n ** 0.5)

    for numero in range(1, n + 1):
        pode = False
        for i, j in product(range(n), repeat=2):
            if tabuleiro[i][j] != 0:
                continue
            if numero in tabuleiro[i,:] or numero in tabuleiro[:,j]:
                continue
            ri, ci = (i//raiz)*raiz, (j//raiz)*raiz
            bloco = tabuleiro[ri:ri+raiz, ci:ci+raiz]
            if numero in bloco:
                continue
            pode = True
            break
        if not pode:
            return True, numero
    return False, None


def simular_jogadas(tabuleiro, pred):
    """
    Simula todas as possíveis jogadas (inserções em células vazias)
    e estima a "solvabilidade" usando o predicado.

    Args:
        tabuleiro (ndarray): Tabuleiro parcialmente preenchido.
        pred (ltn.Predicate): Modelo de predição.

    Returns:
        List[Tuple[(i,j,num), score]]: Jogadas com respectivos scores.
    """
    possibilidades = []
    n = tabuleiro.shape[0]
    vazio = [(i, j) for i, j in product(range(n), repeat=2) if tabuleiro[i][j] == 0]

    for i, j in vazio:
        for num in range(1, n + 1):
            novo = tabuleiro.copy()
            novo[i][j] = num
            tensor = torch.tensor(novo.reshape(1, -1), dtype=torch.float32).to(device)
            score = pred(ltn.Constant(tensor)).value.item()
            possibilidades.append(((i, j, num), score))
    
    return sorted(possibilidades, key=lambda x: x[1], reverse=True)


def avaliar_tabuleiro(tabuleiro, pred):
    """
    Avalia se o tabuleiro tem solução ou está bloqueado.
    Se tiver solução, mostra as jogadas mais promissoras.

    Args:
        tabuleiro (ndarray): Tabuleiro do sudoku.
        pred (ltn.Predicate): Modelo treinado.
    """
    bloqueado, num = tem_num_bloqueado(tabuleiro)
    if bloqueado:
        print(f"Número {num} não pode ser colocado em nenhuma posição.")
        print("Classificação: 1 (Sem solução)")
        return

    print("Classificação: 2 (Solução possível)")
    print("Analisando jogadas com 1 movimento:")
    jogadas = simular_jogadas(tabuleiro, pred)
    for jogada, pontuacao in jogadas[:5]:
        i, j, n = jogada
        print(f"  - Colocar {n} em ({i},{j}) -> Score: {pontuacao:.3f}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        # A mensagem de uso agora reflete a nova flexibilidade
        print("Uso: python questao2.py <pasta_com_csvs_ou_arquivo.csv>")
        exit()

    caminho_entrada = sys.argv[1]
    lista_arquivos_csv = []

    if os.path.isdir(caminho_entrada):
        # Se for um diretório, faz o que já fazia antes
        print(f"Analisando todos os arquivos .csv na pasta: {caminho_entrada}")
        # Cria o caminho completo para cada arquivo
        for arq in os.listdir(caminho_entrada):
            if arq.endswith(".csv"):
                lista_arquivos_csv.append(os.path.join(caminho_entrada, arq))
                
    elif os.path.isfile(caminho_entrada) and caminho_entrada.endswith(".csv"):
        # Se for um único arquivo .csv, adiciona ele à lista
        print(f"Analisando o arquivo: {caminho_entrada}")
        lista_arquivos_csv.append(caminho_entrada)
        
    else:
        # Se não for nem um diretório válido nem um arquivo .csv
        print(f"Erro: O caminho '{caminho_entrada}' não é um diretório válido nem um arquivo .csv.")
        exit()

    predicados = {}

    # O loop principal agora itera sobre a lista de caminhos completos
    for caminho_completo in lista_arquivos_csv:
        print(f"\nArquivo: {os.path.basename(caminho_completo)}")
        tab = pd.read_csv(caminho_completo, header=None).to_numpy()
        tamanho = tab.shape[0]

        if tamanho not in predicados:
            modelo = BoardClassifier(tamanho).to(device)
            pred = ltn.Predicate(modelo)
            treinar_modelo(pred, tamanho)
            predicados[tamanho] = pred
            
        avaliar_tabuleiro(tab, predicados[tamanho])
