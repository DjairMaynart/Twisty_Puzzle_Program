# LLM Puzzle Solver - Guia Rápido

## O que foi implementado

Um solver de cubos mágicos que usa **Ollama** (LLM local) para escolher os próximos movimentos. É o 5º método de resolução, complementando:
- Greedy
- Q-Learning  
- V-Learning
- Neural Network

## Instalação Rápida

### 1. Instalar Ollama
```bash
# Windows: Baixar de https://ollama.ai
# Ou usar winget:
winget install Ollama.Ollama
```

### 2. Baixar um modelo
```bash
ollama pull llama3.2
# Ou modelos menores:
ollama pull llama3.2:1b
ollama pull phi3
```

### 3. Instalar dependência Python
```bash
pip install requests
# Ou atualizar tudo:
pip install -r requirements.txt
```

## Como Usar

### Teste Rápido
```bash
python test_llm_solver.py
```

### Uso no Código

#### Opção 1: Solver Direto (mais simples)
```python
from src.ai_modules.llm_puzzle_solver import solve_with_llm
from src.ai_modules.nn_rl_training import load_puzzle

# Carregar puzzle
solved_state, actions_dict = load_puzzle("cube_2x2x2")

# Criar estado embaralhado (ou usar um existente)
scrambled_state = [...]  # seu estado embaralhado

# Resolver
solution, success = solve_with_llm(
    start_state=scrambled_state,
    ACTIONS_DICT=actions_dict,
    SOLVED_STATE=solved_state,
    max_moves=50,
    ollama_model="llama3.2",
    verbose=True
)
```

#### Opção 2: Integrado com A* (mais eficiente)
```python
from src.puzzle_solver import solve_puzzle
from src.ai_modules.llm_puzzle_solver import LLM_Puzzle_Solver

# Criar solver LLM
llm_solver = LLM_Puzzle_Solver(
    ACTIONS_DICT=actions_dict,
    SOLVED_STATE=solved_state,
    ollama_model="llama3.2"
)

# Usar com A*
solution = solve_puzzle(
    start_state=scrambled_state,
    ACTIONS_DICT=actions_dict,
    SOLVED_STATE=solved_state,
    ai_class=llm_solver,
    max_time=100
)
```

## Como Funciona

1. **Conversão de Estado**: O estado do cubo (lista de inteiros) é convertido para texto descrevendo:
   - Número de posições corretas
   - Distribuição de cores
   - Porcentagem de resolução

2. **Prompt para LLM**: O LLM recebe:
   - Descrição do estado atual
   - Lista de movimentos disponíveis
   - Instrução para sugerir o melhor próximo movimento

3. **Validação**: O movimento sugerido é validado e aplicado. Se inválido, usa fallback greedy.

4. **Cache**: Respostas do LLM são cacheadas para estados idênticos (acelera muito!).

## Vantagens

✅ **Implementação rápida** - Funciona hoje mesmo  
✅ **Sem custo de API** - Ollama roda localmente  
✅ **Fácil de testar** - Basta rodar o script  
✅ **Integrado** - Funciona com o sistema existente  
✅ **Cache inteligente** - Evita chamadas repetidas  

## Limitações

⚠️ **Latência**: Cada movimento requer uma chamada ao LLM (~1-3 segundos)  
⚠️ **Precisão**: LLMs não são perfeitos em raciocínio matemático exato  
⚠️ **Cubos complexos**: Funciona melhor em cubos pequenos (2x2x2, 3x3x3 simples)  

## Dicas

- Use modelos menores (llama3.2:1b) para testes rápidos
- Para cubos maiores, combine com A* (Opção 2)
- Ajuste `max_moves` baseado na complexidade do puzzle
- O cache ajuda muito - estados similares não precisam re-consultar o LLM

## Próximos Passos (Opcional)

- [ ] Fine-tuning do prompt para melhor precisão
- [ ] Suporte a cubos 4D com representação especial
- [ ] Geração de macros/algoritmos pelo LLM
- [ ] Integração com modelos maiores (GPT-4, Claude) via API

## Troubleshooting

**"Ollama is not running"**
- Certifique-se que Ollama está rodando: `ollama serve`
- Verifique se está na porta 11434

**"Model not found"**
- Baixe o modelo: `ollama pull llama3.2`

**Movimentos inválidos**
- O solver tem fallback automático para greedy
- Tente modelos maiores se precisar de mais precisão

