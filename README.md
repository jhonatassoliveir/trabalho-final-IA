# 🧠 Trabalho Final - Inteligência Artificial

**Resolução de Sudoku utilizando SAT Solver, Heurísticas e análise com LTN (Logic Tensor Networks)**

---

## 📁 Estrutura do Projeto

```
TRABALHO_FINAL/
├── questao1.py              # Verificação de solução de Sudoku
├── questao2.py              # Classificação de sudokus com LTN
├── questao3.py              # SAT-solver + Heurísticas + análise LTN
├── sudoku_aberto.csv        # Sudoku 9x9 utilizado na Questão 3
├── sudokus_aberto/
│   └── exemplo_aberto_9x9.csv
└── sudokus_teste/
    └── exemplo_9x9.csv
```

---

## ✅ Questão 1 - Verificação de Solução de Sudoku

### ▶️ Como testar:

```bash
python questao1.py sudokus_teste
```

### 💡 Exemplo de saída:

```
Nota de verificação: 1.0000
Arquivo: exemplo_9x9.csv, Resultado: 1
Resultado 1 indica que o Sudoku está correto.
```

---

## 🧠 Questão 2 - Classificação com Logic Tensor Networks (LTN)

Este modelo LTN classifica se o Sudoku **aberto** é:
- `0`: Sudoku completo
- `1`: Sem solução possível
- `2`: Solução possível

### ▶️ Como testar:

```bash
python questao2.py sudokus_aberto
```

### 💡 Exemplo de saída:

```
Época 0   | Satisfação: 0.421 / 0.424
Época 200 | Satisfação: 1.000 / 0.000
Treinamento finalizado.

Arquivo: exemplo_aberto_9x9.csv
Classificação: 2 (Solução possível)
Analisando jogadas com 1 movimento:
 - Colocar 9 em (8,6) -> Score: 1.000
 - Colocar 9 em (8,4) -> Score: 1.000
...
```

---

## 🔍 Questão 3 - Heurísticas + SAT Solver + Explicação sobre LTN

### ▶️ Como testar:

```bash
python questao3.py
```

> ⚠️ Certifique-se de que o arquivo `sudoku_aberto.csv` esteja na **raiz do projeto**.

### 💡 Exemplo de saída:

```
=== Tabuleiro Carregado ===
[[5 3 0 0 7 0 0 0 0]
 [6 0 0 1 9 5 0 0 0]
 ...]

=== Análise de Heurísticas ===
- Naked Single
- Hidden Single

=== Solução SAT ===
[[5 3 4 6 7 8 9 1 2]
 [6 7 2 1 9 5 3 4 8]
 ...]

=== Sobre a solução com Logic Tensor Networks (LTN) ===
LTNs poderiam ser aplicadas ao Sudoku, mas:
- Requerem mais dados e tempo de treino
- São aproximadas, não exatas
- SAT solvers são mais eficientes e garantem solução
```

---

## 📚 Referências

- Badreddine et al. (2020). *Logic Tensor Networks*.
- Serafini & Garcez (2016). *Logic and Neural Networks*.

---

## 👨‍💻 Autores

- Jhonatas Costa Oliveira  
- Ana Letícia dos Santos Souza  
- Fernanda de Oliveira da Costa  
- Stanley de Carvalho Monteiro  
- Ícaro Costa Moreira


---
