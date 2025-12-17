# LLM Puzzle Solver - Sistema Completo

Sistema de resolução de cubos mágicos usando **LLM (Ollama)** como quinto método de resolução, complementando Greedy, Q-Learning, V-Learning e Neural Network.

## 🚀 Início Rápido

### 1. Instalar Ollama
```bash
# Windows
winget install Ollama.Ollama

# Ou baixar de https://ollama.ai
```

### 2. Baixar Modelo
```bash
ollama pull llama3.2
```

### 3. Instalar Dependências
```bash
pip install -r requirements.txt
```

### 4. Rodar Benchmark
```bash
# Benchmark completo (todos os puzzles)
python run_llm_benchmark.py

# Benchmark rápido (apenas 2x2x2)
python run_quick_benchmark.py
```

## 📊 Resultados

Os resultados são salvos em `benchmark_results/`:
- **JSON**: Dados completos
- **CSV**: Dados tabulares
- **Markdown**: Relatório formatado
- **PNG**: Visualizações (quando disponível)

### Último Benchmark
- Arquivo: `benchmark_20251216_233645.*`
- Visualizações: 6 gráficos de alta qualidade
- Análise estatística: `statistical_analysis.txt`

## 📁 Estrutura

```
src/ai_modules/
├── llm_puzzle_solver.py    # Solver LLM principal
├── llm_benchmark.py        # Sistema de benchmarking
└── visualize_benchmark.py  # Geração de visualizações

run_llm_benchmark.py        # Script principal
run_quick_benchmark.py       # Script rápido
```

## 🎯 Métricas Coletadas

- Movimentos totais até resolver
- Tempo de resolução
- Taxa de sucesso
- Histórico de progresso
- Chamadas LLM e cache hits
- Taxa de progresso (peças/s)
- Velocidade (movimentos/s)

## 📈 Resultados Principais

- **Greedy**: 27.3% taxa de sucesso
- **LLM**: 18.2% taxa de sucesso
- **Hypercuboid 1x1x1x2**: Ambos resolvem com sucesso
- **LLM no 3x3x3**: 251 chamadas em 10 minutos (muito ativo!)

## 🔧 Configuração

Edite `run_llm_benchmark.py` para:
- Alterar puzzles testados
- Ajustar timeouts
- Modificar número de testes
- Escolher modelo Ollama

## 📚 Documentação

- `LLM_SOLVER_README.md` - Guia detalhado do solver
- `BENCHMARK_README.md` - Guia do sistema de benchmarking

