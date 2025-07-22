import os
import pandas as pd
import numpy as np
import torch
import ltn
import sys

# Modelo base para os predicados LTN (sem treinamento necessário)
class BaseModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, *args):
        return None

# Predicado LTN: verifica se uma célula (linha, coluna) contém um número específico
class CellHasValue(ltn.Predicate):
    def __init__(self, model, tensor_board):
        super().__init__(model)
        self.tensor_board = tensor_board
        self.size = tensor_board.shape[0]

    def __call__(self, row, col, value):
        r = torch.round(row).long()
        c = torch.round(col).long()
        v = torch.round(value).long()

        if not ((r >= 0).all() and (r < self.size).all() and
                (c >= 0).all() and (c < self.size).all() and
                (v >= 1).all() and (v <= self.size).all()):
            return torch.tensor(0.0, device=row.device).expand(r.shape)

        return self.tensor_board[r, c, v]

# Predicado LTN: verifica se uma célula está preenchida com exatamente um número
class CellIsFilled(ltn.Predicate):
    def __init__(self, model, tensor_board):
        super().__init__(model)
        self.tensor_board = tensor_board
        self.size = tensor_board.shape[0]

    def __call__(self, row, col):
        r = torch.round(row).long()
        c = torch.round(col).long()
        filled = self.tensor_board[r, c, 1:]
        return (torch.sum(filled, dim=-1) == 1).float()

# Verifica a restrição de linhas: cada número aparece exatamente uma vez por linha
def check_lines(predicate, rows, cols, nums):
    results = []
    for r in rows:
        for n in nums:
            values = [predicate(r.view(1), c.view(1), n.view(1)).squeeze(0) for c in cols]
            total = torch.sum(torch.stack(values))
            results.append((total == 1).float())
    return torch.mean(torch.stack(results)) if results else torch.tensor(0.0)

# Verifica a restrição de colunas: cada número aparece exatamente uma vez por coluna
def check_columns(predicate, rows, cols, nums):
    results = []
    for c in cols:
        for n in nums:
            values = [predicate(r.view(1), c.view(1), n.view(1)).squeeze(0) for r in rows]
            total = torch.sum(torch.stack(values))
            results.append((total == 1).float())
    return torch.mean(torch.stack(results)) if results else torch.tensor(0.0)

# Verifica a restrição de blocos: cada número aparece exatamente uma vez por subgrade
def check_blocks(predicate, rows, cols, nums, block_len):
    n = int(rows[-1].item()) + 1
    if block_len * block_len != n:
        return torch.tensor(0.0)

    results = []
    for i in range(block_len):
        for j in range(block_len):
            for n_val in nums:
                group = []
                for di in range(block_len):
                    for dj in range(block_len):
                        r = torch.tensor(i * block_len + di, dtype=torch.float32)
                        c = torch.tensor(j * block_len + dj, dtype=torch.float32)
                        group.append(predicate(r.view(1), c.view(1), n_val.view(1)).squeeze(0))
                results.append((torch.sum(torch.stack(group)) == 1).float())
    return torch.mean(torch.stack(results)) if results else torch.tensor(0.0)

# Verifica se todas as células do tabuleiro estão preenchidas com apenas um número
def check_all_filled(predicate, rows, cols):
    filled = [predicate(r.view(1), c.view(1)).squeeze(0) for r in rows for c in cols]
    return torch.mean(torch.stack(filled)) if filled else torch.tensor(0.0)

# Função principal para avaliar se um tabuleiro está corretamente preenchido (válido)
def avaliar_sudoku(caminho_csv):
    """
    Avalia um tabuleiro de Sudoku (4x4 ou 9x9) carregado de um arquivo CSV.

    Args:
        caminho_csv (str): Caminho do arquivo CSV contendo o tabuleiro.

    Returns:
        int: 1 se o tabuleiro for válido, 0 caso contrário.
    """
    try:
        df = pd.read_csv(caminho_csv, header=None)
        array_sudoku = df.apply(pd.to_numeric, errors='coerce').to_numpy()
    except Exception as e:
        print(f"Erro ao carregar CSV: {e}")
        return 0

    tamanho = array_sudoku.shape[0]
    if array_sudoku.shape[1] != tamanho or tamanho not in [4, 9]:
        return 0

    if np.isnan(array_sudoku).any():
        return 0

    raiz = int(np.sqrt(tamanho))
    if raiz * raiz != tamanho:
        return 0

    tensor = torch.zeros(tamanho, tamanho, tamanho + 1)
    for i in range(tamanho):
        for j in range(tamanho):
            val = int(array_sudoku[i, j])
            if 1 <= val <= tamanho:
                tensor[i, j, val] = 1

    base_model = BaseModel()
    tem_valor = CellHasValue(base_model, tensor)
    esta_preenchida = CellIsFilled(base_model, tensor)

    linhas = torch.arange(0, tamanho, dtype=torch.float32)
    colunas = torch.arange(0, tamanho, dtype=torch.float32)
    numeros = torch.arange(1, tamanho + 1, dtype=torch.float32)

    try:
        l = check_lines(tem_valor, linhas, colunas, numeros)
        c = check_columns(tem_valor, linhas, colunas, numeros)
        b = check_blocks(tem_valor, linhas, colunas, numeros, raiz)
        p = check_all_filled(esta_preenchida, linhas, colunas)
        nota_geral = torch.mean(torch.stack([l, c, b, p]))
    except Exception as e:
        print(f"Erro na avaliação: {e}")
        return 0

    print(f"Nota de verificação: {nota_geral.item():.4f}")
    return 1 if nota_geral >= 0.99 else 0

# Execução em modo script para ler vários arquivos CSV
if __name__ == "__main__":
    if len(sys.argv) > 1:
        pasta = sys.argv[1]
        for arquivo in os.listdir(pasta):
            if arquivo.endswith(".csv"):
                caminho = os.path.join(pasta, arquivo)
                resultado = avaliar_sudoku(caminho)
                print(f"Arquivo: {arquivo}, Resultado: {resultado}")
    else:
        print("Uso: python nome_do_script.py <caminho_para_pasta_com_csvs>")
