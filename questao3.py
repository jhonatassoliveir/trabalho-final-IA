import numpy as np
import pandas as pd
from itertools import product, combinations
from pysat.formula import CNF
from pysat.solvers import Solver


class SudokuSolver:
    """
    Classe responsável por resolver tabuleiros de Sudoku utilizando SAT solver.
    Inclui validações, construção de cláusulas lógicas (CNF), análise heurística e explicação sobre LTN.
    """

    def __init__(self, tamanho=9):
        """Inicializa o solver com o tamanho do tabuleiro."""
        self.n = tamanho
        self.bloco = int(tamanho ** 0.5)
        self.cnf = CNF()
        self.var_map = {}

    def _var(self, r, c, v):
        """Mapeia uma variável (r, c, v) para um número inteiro único."""
        if (r, c, v) not in self.var_map:
            self.var_map[(r, c, v)] = len(self.var_map) + 1
        return self.var_map[(r, c, v)]

    def carregar_sudoku_csv(self, caminho):
        """Carrega o tabuleiro de um arquivo CSV."""
        df = pd.read_csv(caminho, header=None)
        tabuleiro = df.to_numpy()

        if tabuleiro.shape[0] != tabuleiro.shape[1]:
            raise ValueError("Tabuleiro deve ser quadrado")

        self.n = tabuleiro.shape[0]
        self.bloco = int(self.n ** 0.5)
        return tabuleiro.astype(int)

    def validar_tabuleiro(self, tabuleiro):
        """Verifica se o tabuleiro inicial não viola as regras do Sudoku."""
        if np.any(tabuleiro < 0) or np.any(tabuleiro > self.n):
            return False

        # Verifica linhas e colunas
        for i in range(self.n):
            if len(set(tabuleiro[i, tabuleiro[i, :] != 0])) != len(tabuleiro[i, tabuleiro[i, :] != 0]):
                return False
            if len(set(tabuleiro[tabuleiro[:, i] != 0, i])) != len(tabuleiro[tabuleiro[:, i] != 0, i]):
                return False

        # Verifica blocos
        for bi, bj in product(range(self.bloco), repeat=2):
            bloco = tabuleiro[
                bi * self.bloco:(bi + 1) * self.bloco,
                bj * self.bloco:(bj + 1) * self.bloco
            ]
            numeros = bloco[bloco != 0]
            if len(numeros) != len(set(numeros)):
                return False

        return True

    def construir_cnf(self, tabuleiro):
        """Gera as cláusulas CNF baseadas no tabuleiro atual."""
        self.cnf = CNF()
        self.var_map = {}

        # Restrições de valor único por célula
        for r, c in product(range(self.n), repeat=2):
            if tabuleiro[r][c] == 0:
                self.cnf.append([self._var(r, c, v) for v in range(self.n)])
            else:
                v = tabuleiro[r][c] - 1
                self.cnf.append([self._var(r, c, v)])

        # Restrições para impedir múltiplos valores na mesma célula
        for r, c in product(range(self.n), repeat=2):
            for v1, v2 in combinations(range(self.n), 2):
                self.cnf.append([-self._var(r, c, v1), -self._var(r, c, v2)])

        # Cada valor aparece no máximo uma vez por linha
        for r, v in product(range(self.n), range(self.n)):
            self.cnf.append([self._var(r, c, v) for c in range(self.n)])
            for c1, c2 in combinations(range(self.n), 2):
                self.cnf.append([-self._var(r, c1, v), -self._var(r, c2, v)])

        # Cada valor aparece no máximo uma vez por coluna
        for c, v in product(range(self.n), range(self.n)):
            self.cnf.append([self._var(r, c, v) for r in range(self.n)])
            for r1, r2 in combinations(range(self.n), 2):
                self.cnf.append([-self._var(r1, c, v), -self._var(r2, c, v)])

        # Cada valor aparece no máximo uma vez por bloco
        for bi, bj, v in product(range(self.bloco), range(self.bloco), range(self.n)):
            celulas = [
                self._var(bi * self.bloco + i, bj * self.bloco + j, v)
                for i, j in product(range(self.bloco), repeat=2)
            ]
            self.cnf.append(celulas)
            for i, j in combinations(range(len(celulas)), 2):
                self.cnf.append([-celulas[i], -celulas[j]])

    def resolver_sat(self, tabuleiro):
        """Resolve o tabuleiro com base nas cláusulas geradas."""
        if not self.validar_tabuleiro(tabuleiro):
            print("Tabuleiro inválido.")
            return None

        self.construir_cnf(tabuleiro)
        solver = Solver(bootstrap_with=self.cnf)

        if solver.solve():
            modelo = solver.get_model()
            solucao = np.zeros((self.n, self.n), dtype=int)
            for (r, c, v), var in self.var_map.items():
                if var in modelo:
                    solucao[r][c] = v + 1
            return solucao
        return None

    def analisar_heuristicas(self, tabuleiro):
        """Aplica heurísticas básicas de resolução ao tabuleiro."""
        heuristicas = {
            "Naked Single": False,
            "Hidden Single": False,
            "Locked Candidates": False,
            "Naked Pair": False,
            "X-Wing": False
        }

        for r in range(self.n):
            for c in range(self.n):
                if tabuleiro[r, c] == 0:
                    candidatos = [v for v in range(1, self.n + 1)
                                  if self.pode_colocar(tabuleiro, r, c, v)]
                    if len(candidatos) == 1:
                        heuristicas["Naked Single"] = True

        for v in range(1, self.n + 1):
            for r in range(self.n):
                cols = [c for c in range(self.n)
                        if tabuleiro[r, c] == 0 and self.pode_colocar(tabuleiro, r, c, v)]
                if len(cols) == 1:
                    heuristicas["Hidden Single"] = True

            for c in range(self.n):
                rows = [r for r in range(self.n)
                        if tabuleiro[r, c] == 0 and self.pode_colocar(tabuleiro, r, c, v)]
                if len(rows) == 1:
                    heuristicas["Hidden Single"] = True

        print("\nHeurísticas recomendadas:")
        for h, ativa in heuristicas.items():
            if ativa:
                print(f"- {h}")
        return heuristicas

    def pode_colocar(self, tabuleiro, r, c, v):
        """Verifica se o número v pode ser colocado na posição (r,c)."""
        if v in tabuleiro[r, :] or v in tabuleiro[:, c]:
            return False
        br, bc = r // self.bloco, c // self.bloco
        bloco = tabuleiro[br * self.bloco:(br + 1) * self.bloco,
                          bc * self.bloco:(bc + 1) * self.bloco]
        return v not in bloco

    def explicacao_ltn(self):
        """Imprime explicações sobre a aplicação (ou não) de LTNs para Sudoku."""
        print("\n=== Sobre a solução com Logic Tensor Networks (LTN) ===")
        print("Teoricamente, sim: LTNs poderiam ser aplicadas para resolver Sudoku,")
        print("pois são redes neurais que combinam lógica simbólica com aprendizado profundo.")
        print("\nNo entanto, na prática:")
        print("- LTNs são mais complexas para implementar que métodos tradicionais")
        print("- O desempenho geralmente é inferior a algoritmos dedicados como SAT")
        print("- Requerem grande quantidade de dados e tempo de treinamento")
        print("- Resultados são aproximados, não exatos como métodos lógicos")
        print("\nEste solver optou por manter apenas a implementação SAT, que:")
        print("- Garante soluções exatas")
        print("- É computacionalmente eficiente")
        print("- Tem implementação robusta e testada")
        print("\nReferências sobre LTN:")
        print("- Badreddine et al. (2020): Logic Tensor Networks")
        print("- Serafini & Garcez (2016): Logic and Neural Networks")


def testar_sudoku():
    """Função principal para testar o solver com um arquivo CSV."""
    nome_arquivo = "sudoku_aberto.csv"
    solver = SudokuSolver()

    try:
        tabuleiro = solver.carregar_sudoku_csv(nome_arquivo)
        print("\n=== Tabuleiro Carregado ===")
        print(tabuleiro)

        if not solver.validar_tabuleiro(tabuleiro):
            print("Tabuleiro inválido!")
            return

        print("\n=== Análise de Heurísticas ===")
        solver.analisar_heuristicas(tabuleiro)

        print("\n=== Solução SAT ===")
        sol_sat = solver.resolver_sat(tabuleiro)
        print(sol_sat)

        solver.explicacao_ltn()

    except Exception as e:
        print(f"Erro: {str(e)}")


if __name__ == "__main__":
    testar_sudoku()
