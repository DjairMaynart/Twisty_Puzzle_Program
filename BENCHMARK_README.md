# Sistema de Benchmarking - LLM Puzzle Solver

Sistema completo de benchmarking, métricas e comparação de desempenho para o LLM Puzzle Solver.

## 🎯 Funcionalidades

- ✅ **Benchmarking Automatizado**: Testa múltiplos puzzles e métodos
- ✅ **Métricas Detalhadas**: Movimentos, tempo, chamadas LLM, cache hits
- ✅ **Comparação de Métodos**: LLM vs Greedy vs Q-Learning vs V-Learning
- ✅ **Relatórios Múltiplos**: JSON, CSV, Markdown
- ✅ **Visualizações**: Gráficos comparativos e análises
- ✅ **Prompts Inteligentes**: Adaptados para diferentes tipos de puzzles

## 📊 Métricas Coletadas

Para cada teste, o sistema coleta:
- **Taxa de sucesso**: % de puzzles resolvidos
- **Número de movimentos**: Média, mediana, min, max
- **Tempo de resolução**: Em segundos
- **Estados visitados**: Quantos estados foram explorados
- **Chamadas LLM**: Total de queries ao Ollama (apenas LLM)
- **Cache hits**: Quantas respostas vieram do cache (apenas LLM)
- **Taxa de cache**: % de cache hits vs total de queries

## 🚀 Como Usar

### Benchmark Rápido (Validação)

```bash
python run_quick_benchmark.py
```

Testa apenas o cubo 2x2x2 com 2 testes por método. Ideal para validação rápida.

### Benchmark Completo

```bash
python run_llm_benchmark.py
```

Executa benchmark completo com:
- **cube_2x2x2**: 3 testes, scramble de 8 movimentos
- **cube_3x3x3**: 2 testes, scramble de 10 movimentos  
- **hypercuboid_1x1x1x2**: 3 testes, scramble de 6 movimentos

### Gerar Visualizações

```bash
python -c "from src.ai_modules.visualize_benchmark import generate_all_visualizations; generate_all_visualizations('benchmark_results/benchmark_YYYYMMDD_HHMMSS.json')"
```

Ou simplesmente:
```bash
python src/ai_modules/visualize_benchmark.py
```

Isso gera automaticamente gráficos do JSON mais recente.

## 📁 Estrutura de Arquivos

```
benchmark_results/
├── benchmark_YYYYMMDD_HHMMSS.json      # Dados completos (JSON)
├── benchmark_YYYYMMDD_HHMMSS.csv       # Dados tabulares (CSV)
├── benchmark_YYYYMMDD_HHMMSS.md        # Relatório formatado (Markdown)
├── comparison.png                       # Comparação geral de métodos
├── by_puzzle.png                       # Performance por puzzle
└── llm_metrics.png                     # Métricas específicas do LLM
```

## 📈 Relatórios Gerados

### JSON
Dados completos em formato estruturado, ideal para análise programática.

### CSV
Dados tabulares para importação em Excel, Google Sheets, etc.

### Markdown
Relatório formatado com:
- Estatísticas por puzzle e método
- Tabelas comparativas
- Taxas de sucesso
- Métricas agregadas

### Visualizações PNG

1. **comparison.png**: Comparação geral entre métodos
   - Taxa de sucesso
   - Movimentos médios
   - Tempo médio
   - Scatter plot movimentos vs tempo

2. **by_puzzle.png**: Performance por puzzle
   - Movimentos por puzzle
   - Tempo por puzzle

3. **llm_metrics.png**: Métricas específicas do LLM
   - Chamadas LLM vs Movimentos
   - Eficiência do cache
   - Tempo por chamada
   - Eficiência geral

## 🔧 Configuração

### Ajustar Puzzles Testados

Edite `run_llm_benchmark.py`:

```python
puzzles_to_test = [
    {
        "name": "cube_2x2x2",
        "scramble_length": 8,
        "max_moves": 50,
        "num_tests": 3,
    },
    # Adicione mais puzzles aqui
]
```

### Ajustar Métodos

```python
methods = ["llm", "greedy"]  # Adicione "q", "v", "nn" quando disponíveis
```

### Ajustar Modelo LLM

```python
ollama_model="llama3.2"  # Mude para outro modelo se necessário
```

## 📊 Exemplo de Saída

```
======================================================================
Benchmark: cube_2x2x2
======================================================================
Puzzle carregado: 24 stickers, 12 movimentos

--- Teste 1/3 ---
Estado embaralhado: 8/24 corretos (33.3%)
  Testando LLM... SUCESSO (12 movimentos, 15.23s)
  Testando GREEDY... SUCESSO (18 movimentos, 0.05s)

--- Teste 2/3 ---
Estado embaralhado: 10/24 corretos (41.7%)
  Testando LLM... SUCESSO (10 movimentos, 12.45s)
  Testando GREEDY... SUCESSO (15 movimentos, 0.04s)
```

## 🎨 Prompts Inteligentes

O sistema adapta os prompts do LLM baseado no tipo de puzzle:

- **2x2x2**: Foca em orientação de cantos
- **3x3x3**: Estratégia camada por camada
- **4D/Hyper**: Instruções específicas para puzzles 4D
- **Outros**: Estratégias gerais

## 🔍 Análise de Resultados

### Interpretando Métricas

- **Taxa de sucesso**: % de puzzles resolvidos com sucesso
- **Movimentos**: Menor é melhor (soluções mais eficientes)
- **Tempo**: Depende do método (LLM é mais lento mas pode ser mais inteligente)
- **Cache hits**: Mostra eficiência do cache (maior = menos chamadas ao LLM)

### Comparação LLM vs Greedy

- **LLM**: Geralmente mais movimentos, mas pode resolver puzzles mais complexos
- **Greedy**: Rápido, mas limitado em puzzles complexos
- **Cache**: Reduz drasticamente o tempo do LLM em testes repetidos

## 🐛 Troubleshooting

**Ollama não conecta**: Certifique-se que `ollama serve` está rodando

**Puzzle não encontrado**: Verifique se o puzzle existe em `src/puzzles/`

**Erro de importação**: Instale dependências: `pip install -r requirements.txt`

**Visualizações não geram**: Instale matplotlib: `pip install matplotlib pandas`

## 📝 Notas

- O benchmark pode demorar bastante tempo (especialmente 3x3x3)
- Cada chamada ao LLM leva ~1-3 segundos
- O cache acelera muito testes subsequentes
- Para benchmarks longos, considere usar modelos menores (llama3.2:1b)

## 🎯 Próximos Passos

- [ ] Integração com Q-Learning e V-Learning
- [ ] Suporte para Neural Networks
- [ ] Benchmark distribuído (múltiplos puzzles em paralelo)
- [ ] Análise estatística avançada
- [ ] Exportação para LaTeX (tabelas de artigos)

