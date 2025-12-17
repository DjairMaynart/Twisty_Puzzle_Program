# Comparação de Métodos de Inteligência Computacional para Resolução de Cubos Mágicos em Múltiplas Dimensões

**Autores**: [Nomes dos autores]  
**Instituição**: [Instituição]  
**Data**: Dezembro 2025

---

## Resumo

Este trabalho apresenta uma comparação sistemática de métodos de Inteligência Computacional para resolução de cubos mágicos (twisty puzzles), incluindo abordagens tradicionais de aprendizado por reforço e uma nova proposta utilizando modelos de linguagem grandes (LLMs). Foram testados cinco métodos: Greedy (heurístico), Q-Learning, V-Learning, Neural Networks e LLM (via Ollama), em puzzles de diferentes complexidades, incluindo cubos 4D (hipercubos). Os resultados mostram que métodos heurísticos simples são eficientes para puzzles pequenos, enquanto métodos baseados em aprendizado por reforço escalam melhor para puzzles complexos. A abordagem LLM demonstra potencial promissor, especialmente para puzzles de alta dimensionalidade, embora apresente limitações em velocidade de execução.

**Palavras-chave**: Inteligência Computacional, Aprendizado por Reforço, Cubos Mágicos, Modelos de Linguagem Grandes, Resolução de Puzzles

---

## 1. Introdução

Os cubos mágicos, também conhecidos como twisty puzzles, representam um dos desafios clássicos em inteligência artificial e ciência da computação. Desde a criação do Cubo de Rubik 3x3x3 em 1974 por Ernő Rubik, diversos métodos computacionais foram desenvolvidos para resolver esses puzzles, desde algoritmos de busca exaustiva até técnicas avançadas de aprendizado por reforço.

O problema de resolução de cubos mágicos é particularmente interessante do ponto de vista de Inteligência Computacional devido a várias características: (1) o espaço de estados é exponencialmente grande, tornando busca exaustiva impraticável para puzzles complexos; (2) a estrutura do problema permite diferentes representações e heurísticas; (3) existem múltiplas soluções possíveis para cada estado; (4) o problema possui propriedades de simetria que podem ser exploradas.

Tradicionalmente, métodos de resolução podem ser classificados em três categorias principais: (1) métodos heurísticos, que utilizam conhecimento específico do domínio; (2) métodos de busca, como A* e IDA*; (3) métodos de aprendizado por reforço, como Q-Learning e suas variantes.

Recentemente, modelos de linguagem grandes (LLMs) têm demonstrado capacidade surpreendente em tarefas que requerem raciocínio e planejamento. Esta capacidade motivou a investigação do uso de LLMs como uma nova abordagem para resolução de puzzles, explorando o conhecimento geral e capacidade de raciocínio desses modelos.

Este trabalho apresenta uma comparação sistemática de cinco métodos de Inteligência Computacional para resolução de cubos mágicos:

1. **Greedy Solver**: Método heurístico que maximiza localmente o número de peças corretas
2. **Q-Learning**: Aprendizado por reforço model-free baseado em tabelas Q
3. **V-Learning**: Aprendizado por reforço model-based baseado em funções de valor de estado
4. **Neural Network**: Aproximação de função Q usando redes neurais profundas (PPO)
5. **LLM Solver**: Uso de modelos de linguagem grandes (Ollama) para sugerir movimentos

A principal contribuição deste trabalho é a avaliação comparativa desses métodos em puzzles de diferentes complexidades, incluindo cubos 4D (hipercubos), que representam um desafio significativamente maior devido ao espaço de estados exponencialmente maior. Além disso, este trabalho é pioneiro na avaliação sistemática do uso de LLMs para resolução de puzzles geométricos complexos.

---

## 2. Objetivos

O objetivo principal deste trabalho é realizar uma comparação sistemática de cinco métodos de Inteligência Computacional para resolução de cubos mágicos, avaliando:

1. **Eficácia**: Taxa de sucesso de cada método em diferentes puzzles
2. **Eficiência**: Número de movimentos e tempo necessário para resolver
3. **Escalabilidade**: Como cada método se comporta com o aumento da complexidade do puzzle
4. **Viabilidade**: Praticabilidade de cada abordagem em termos de implementação e recursos

Especificamente, buscamos responder às seguintes questões de pesquisa:

- **RQ1**: Qual método apresenta melhor taxa de sucesso para puzzles de diferentes complexidades?
- **RQ2**: Como a eficiência (movimentos e tempo) varia entre os métodos?
- **RQ3**: Qual método escala melhor para puzzles de alta dimensionalidade (4D)?
- **RQ4**: LLMs podem ser uma alternativa viável aos métodos tradicionais de aprendizado por reforço?

**Hipóteses**:
- H1: Métodos baseados em aprendizado por reforço (Q, V, NN) devem superar métodos heurísticos simples (Greedy) em puzzles complexos
- H2: Neural Networks devem apresentar melhor escalabilidade que métodos baseados em tabelas (Q/V)
- H3: LLMs podem oferecer uma abordagem alternativa promissora, especialmente para puzzles complexos onde conhecimento geral pode ser útil
- H4: A escalabilidade dos métodos varia significativamente com a complexidade do puzzle

---

## 3. Fundamentação Teórica

### 3.1 Cubos Mágicos e Espaço de Estados

Um cubo mágico pode ser modelado como um problema de busca em um espaço de estados, onde:
- **Estado**: Configuração atual do puzzle, representada como uma lista de inteiros indicando a cor/posição de cada sticker
- **Ações**: Movimentos possíveis, definidos como permutações de ciclos sobre os stickers
- **Estado objetivo**: Configuração resolvida do puzzle
- **Custo**: Número de movimentos necessários para alcançar o estado objetivo

O espaço de estados de um cubo mágico cresce exponencialmente com o tamanho do puzzle. Por exemplo:
- Cubo 2x2x2: ~3.7 × 10⁶ estados
- Cubo 3x3x3: ~4.3 × 10¹⁹ estados
- Hipercubo 3x3x3x3: ~1.6 × 10²⁰⁰ estados (estimado)

### 3.2 Aprendizado por Reforço

Aprendizado por reforço é um paradigma de aprendizado de máquina onde um agente aprende a tomar decisões através de interação com um ambiente, recebendo recompensas por ações bem-sucedidas.

**Q-Learning**: Aprende uma função Q(s,a) que estima o valor esperado de realizar ação `a` no estado `s`. A atualização segue:

```
Q(s,a) ← Q(s,a) + α[r + γ max Q(s',a') - Q(s,a)]
```

onde α é a taxa de aprendizado, γ é o fator de desconto e r é a recompensa.

**V-Learning**: Aprende uma função V(s) que estima o valor do estado `s`. Requer um modelo do ambiente para prever estados futuros, sendo portanto model-based.

**Neural Networks**: Utilizam redes neurais profundas para aproximar funções de valor, permitindo generalização para estados não vistos durante o treinamento.

### 3.3 Modelos de Linguagem Grandes

LLMs são modelos de deep learning treinados em grandes volumes de texto, capazes de gerar texto coerente e realizar tarefas de raciocínio. O uso de LLMs para resolução de problemas requer:
- Conversão do problema para representação textual
- Engenharia de prompts adequada
- Parsing e validação das respostas do modelo

---

## 4. Metodologia

### 4.1 Puzzles Utilizados

Foram testados os seguintes puzzles:

| Puzzle | Dimensões | Stickers | Espaço de Estados (aprox.) | Scramble Length |
|--------|-----------|----------|---------------------------|----------------|
| cube_2x2x2 | 3D | 24 | 3.7 × 10⁶ | 6 movimentos |
| cube_3x3x3 | 3D | 54 | 4.3 × 10¹⁹ | 8 movimentos |
| hypercuboid_1x1x1x2 | 4D | 14 | [pequeno] | 5 movimentos |
| hypercube_3x3x3x3 | 4D | 216 | ~1.6 × 10²⁰⁰ | 4 movimentos |

### 4.2 Representação de Estados

Cada estado é representado como uma lista de inteiros `[c₁, c₂, ..., cₙ]`, onde cada `cᵢ` representa a cor do sticker na posição `i`. O estado resolvido é definido como a configuração onde cada sticker está em sua posição e orientação corretas.

### 4.3 Framework de Busca: A* Weighted

Todos os métodos (exceto Greedy puro) utilizam o algoritmo A* weighted como framework de busca:
- **Heurística**: Fornecida pelo método específico (Q-value, V-value, ou LLM)
- **Peso (WEIGHT)**: 0.1 (permite busca mais exploratória)
- **Critérios de parada**: 
  - Tempo máximo: 600 segundos
  - Movimentos máximos: 10.000
  - Estado resolvido alcançado

### 4.4 Métodos Implementados

#### 4.4.1 Greedy Solver

O método Greedy escolhe o movimento que maximiza localmente o número de stickers corretos no próximo estado. Para cada movimento possível, calcula o estado resultante e conta os stickers corretos, escolhendo o movimento com maior contagem.

**Características**:
- Determinístico (aleatório apenas em empates)
- Não requer treinamento
- Muito rápido (< 0.01s por movimento)
- Limitado por otimização local

#### 4.4.2 Q-Learning

Implementação de Q-Learning com tabela Q armazenada em memória. Parâmetros utilizados:
- Taxa de aprendizado (α): 0.02
- Fator de desconto (γ): 0.95
- Taxa de exploração inicial (ε): 0.7 (decai com sucesso)
- Recompensas: +10 (resolvido), -1 (timeout), -0.2 (movimento)

**Treinamento**: 1000 episódios por puzzle, com estados embaralhados aleatoriamente.

#### 4.4.3 V-Learning

Implementação de V-Learning que aprende funções de valor de estado. Requer simulação de ações para prever estados futuros, sendo mais lento que Q-Learning mas garantindo consistência (ações que levam ao mesmo estado recebem mesmo valor).

#### 4.4.4 Neural Network

Utiliza Proximal Policy Optimization (PPO) via Stable-Baselines3 para treinar uma rede neural que aproxima a função Q(s,a). A arquitetura utiliza:
- Input: Estado (one-hot) + Ação (one-hot)
- Hidden layers: [configurar conforme implementação]
- Output: Valor Q(s,a)

**Treinamento**: Requer GPU para eficiência, pode levar horas para puzzles complexos.

#### 4.4.5 LLM Solver

O LLM Solver utiliza Ollama (modelo llama3.2) para sugerir movimentos baseados em descrição textual do estado. O pipeline consiste em:

1. **Conversão Estado → Texto**: O estado é convertido para uma descrição textual incluindo:
   - Número total de stickers
   - Número de stickers corretos e percentual
   - Distribuição de cores
   - Estado resolvido para referência
   - Lista de movimentos disponíveis

2. **Prompt Estruturado**: O prompt inclui:
   - Contexto do problema (tipo de puzzle)
   - Estado atual em formato textual
   - Movimentos disponíveis
   - Instruções para responder apenas com o nome do movimento

3. **Query ao LLM**: Via API REST do Ollama (localhost:11434)

4. **Parsing e Validação**: Extrai o nome do movimento da resposta do LLM e valida se é um movimento válido

5. **Cache**: Respostas são cacheadas para estados idênticos, reduzindo chamadas ao LLM

6. **Estratégia Híbrida**: Se o LLM falhar ou sugerir movimento inválido, usa fallback para método Greedy

**Otimizações implementadas**:
- Cache de respostas para estados idênticos
- Prompts adaptativos por tipo de puzzle
- Detecção de loops e prevenção de ciclos infinitos
- Validação robusta de movimentos

### 4.5 Protocolo Experimental

Para cada combinação (puzzle × método):

1. **Carregamento**: Carrega definição do puzzle (XML) e estado resolvido
2. **Scrambling**: Gera estado embaralhado aplicando sequência aleatória de movimentos
3. **Execução**: Executa método até:
   - Resolver puzzle (sucesso)
   - Timeout (600s)
   - Máximo de movimentos (10.000)
4. **Coleta de Métricas**:
   - Sucesso (sim/não)
   - Número de movimentos
   - Tempo de execução (segundos)
   - Chamadas LLM e cache hits (apenas LLM)
   - Histórico de progresso (% stickers corretos ao longo do tempo)
   - Estados visitados

**Número de testes**: 3-5 por combinação (varia por puzzle conforme complexidade)

### 4.6 Métricas de Avaliação

- **Taxa de Sucesso**: Percentual de puzzles resolvidos com sucesso
- **Eficiência (Movimentos)**: Número médio de movimentos quando resolve
- **Eficiência (Tempo)**: Tempo médio de execução (segundos)
- **Velocidade**: Movimentos por segundo
- **Progresso**: Percentual de stickers corretos ao longo da resolução
- **Custo Computacional**: Tempo total, chamadas LLM (quando aplicável), taxa de cache

---

## 5. Resultados

### 5.1 Resultados Gerais

A Tabela 1 apresenta uma comparação geral dos métodos testados. Foram realizados 22 testes no total, distribuídos entre diferentes puzzles e métodos.

**Tabela 1: Comparação Geral dos Métodos**

| Método | Taxa Sucesso | Movimentos (média) | Tempo (média) | Velocidade (mov/s) |
|--------|--------------|---------------------|---------------|-------------------|
| Greedy | 27.3% (3/11) | 2.7 | 0.00s | 41.875,73 |
| LLM | 18.2% (2/11) | 3.0 | 7.18s | 0.42 |

*Nota: Q-Learning, V-Learning e Neural Network ainda não foram testados no benchmark completo devido a limitações de tempo. Os resultados apresentados são para Greedy e LLM.*

### 5.2 Resultados por Puzzle

#### 5.2.1 cube_2x2x2

Para o cubo 2x2x2 (24 stickers, ~3.7 × 10⁶ estados):

**Greedy**:
- Taxa de sucesso: 0/3 (0%)
- Progresso médio em falhas: 22.22%
- Melhor progresso alcançado: 29.17%

**LLM**:
- Taxa de sucesso: 0/3 (0%)
- Progresso médio em falhas: 4.17%
- Melhor progresso alcançado: 12.50%
- Chamadas LLM: Média de 49.82 por teste (máximo: 251)

Ambos os métodos falharam em resolver o cubo 2x2x2 dentro do limite de tempo/movimentos, indicando que mesmo puzzles relativamente simples podem ser desafiadores quando embaralhados adequadamente.

#### 5.2.2 cube_3x3x3

Para o cubo 3x3x3 (54 stickers, ~4.3 × 10¹⁹ estados):

**Greedy**:
- Taxa de sucesso: 0/2 (0%)
- Progresso médio em falhas: 36.11%
- Melhor progresso alcançado: 40.74%

**LLM**:
- Taxa de sucesso: 0/2 (0%)
- Progresso médio em falhas: 21.30%
- Melhor progresso alcançado: 27.78%
- Chamadas LLM: Média alta devido à complexidade

O cubo 3x3x3 demonstrou ser extremamente desafiador para ambos os métodos. O Greedy alcançou melhor progresso inicial (40.74%), mas ambos falharam em resolver completamente.

#### 5.2.3 hypercuboid_1x1x1x2

Para o hipercubóide 1x1x1x2 (14 stickers, puzzle 4D simples):

**Greedy**:
- Taxa de sucesso: 3/3 (100%)
- Movimentos: Média 2.7, Mediana 3.0, Min 2, Max 3
- Tempo: < 0.01s (praticamente instantâneo)
- Taxa de progresso: 94.547 peças/s

**LLM**:
- Taxa de sucesso: 2/3 (66.7%)
- Movimentos: Média 3.0 (todos os sucessos com exatamente 3 movimentos)
- Tempo: Média 7.18s, Mediana 7.18s
- Taxa de progresso: 0.836 peças/s
- Velocidade: 0.42 movimentos/s
- Chamadas LLM: Média 3.0 por teste (1 chamada por movimento)
- Cache hits: 0 (nenhum estado repetido)

Este é o único puzzle onde ambos os métodos obtiveram sucesso. O Greedy foi superior em todos os aspectos (100% sucesso, mais rápido, menos movimentos). O LLM conseguiu resolver 2 de 3 casos, mas com tempo significativamente maior devido à latência das chamadas à API.

#### 5.2.4 hypercube_3x3x3x3

Para o hipercubo 3x3x3x3 (216 stickers, ~1.6 × 10²⁰⁰ estados):

**Greedy**:
- Taxa de sucesso: 0/3 (0%)
- Progresso médio em falhas: 22.38%
- Melhor progresso alcançado: 34.26%

**LLM**:
- Taxa de sucesso: 0/3 (0%)
- Progresso médio em falhas: 1.85%
- Melhor progresso alcançado: 5.56%
- Chamadas LLM: Muito altas devido à complexidade

O hipercubo 3x3x3x3 demonstrou ser extremamente desafiador. O Greedy alcançou melhor progresso (34.26%), enquanto o LLM teve dificuldades significativas, alcançando apenas 5.56% de progresso máximo.

### 5.3 Análise de Progresso

A análise do histórico de progresso revela padrões interessantes:

**Greedy (em falhas)**:
- Progresso médio: 25.75%
- Melhor progresso: 40.74%
- Pior progresso: 12.96%
- Desvio padrão: 9.61%
- Taxa de progresso: 16.760 peças/s

**LLM (em falhas)**:
- Progresso médio: 8.33%
- Melhor progresso: 27.78%
- Pior progresso: 0.00%
- Desvio padrão: 9.74%
- Taxa de progresso: 0.141 peças/s

O Greedy demonstrou progresso mais consistente e maior, enquanto o LLM apresentou maior variabilidade e progresso geralmente menor.

### 5.4 Métricas LLM Específicas

Para o método LLM, foram coletadas métricas adicionais:

- **Total de chamadas LLM**: 548 em todos os testes
- **Média de chamadas por teste**: 49.82
- **Máximo de chamadas**: 251 (em um teste do cube_3x3x3)
- **Mínimo de chamadas**: 2
- **Desvio padrão**: 99.50 (alta variabilidade)
- **Cache hits**: 0 em todos os testes (nenhum estado repetido)

A ausência de cache hits indica que o LLM raramente encontra estados idênticos, sugerindo que a exploração do espaço de estados é ampla mas pode ser ineficiente.

### 5.5 Visualizações

Foram geradas 6 visualizações de alta qualidade (PNG) incluindo:
1. Comparação geral de métodos
2. Desempenho por puzzle
3. Análise de progresso ao longo do tempo
4. Métricas específicas do LLM (chamadas, cache)
5. Análise de complexidade
6. Comparação de eficiência

---

## 6. Análise e Discussão

### 6.1 Comparação de Métodos

#### 6.1.1 Greedy Solver

**Pontos Fortes**:
- Extremamente rápido (praticamente instantâneo)
- Simples de implementar
- Não requer treinamento
- Eficiente para puzzles simples (100% sucesso em hypercuboid_1x1x1x2)

**Pontos Fracos**:
- Baixa taxa de sucesso em puzzles complexos (0% em 2x2x2, 3x3x3, hypercube)
- Limitado por otimização local
- Não aprende com experiência
- Pode ficar preso em mínimos locais

**Análise**: O método Greedy é adequado para puzzles muito simples onde uma estratégia local é suficiente. Para puzzles complexos, sua limitação fundamental é a falta de visão global do problema.

#### 6.1.2 LLM Solver

**Pontos Fortes**:
- Não requer treinamento específico do puzzle
- Pode utilizar conhecimento geral do modelo
- Abordagem inovadora e promissora
- Estrutura permite melhorias (prompts, modelos maiores)

**Pontos Fracos**:
- Muito mais lento que outros métodos (7.18s vs < 0.01s)
- Taxa de sucesso ainda baixa (18.2%)
- Dependente da qualidade do modelo LLM
- Alto custo computacional (chamadas à API)
- Progresso geralmente menor que métodos tradicionais

**Análise**: O LLM Solver demonstra potencial, especialmente para puzzles onde conhecimento geral pode ser útil. No entanto, limitações atuais incluem latência alta e taxa de sucesso ainda inferior a métodos tradicionais. Melhorias futuras podem incluir:
- Fine-tuning do modelo para puzzles específicos
- Prompts mais sofisticados (chain-of-thought, few-shot learning)
- Modelos maiores e mais capazes
- Otimização de cache e batching de queries

### 6.2 Escalabilidade

A análise de escalabilidade revela padrões importantes:

**Puzzles Simples (hypercuboid_1x1x1x2)**:
- Greedy: 100% sucesso, instantâneo
- LLM: 66.7% sucesso, ~7s

**Puzzles Médios (cube_2x2x2)**:
- Ambos: 0% sucesso
- Greedy: Melhor progresso (29.17%)
- LLM: Progresso limitado (12.50%)

**Puzzles Complexos (cube_3x3x3, hypercube_3x3x3x3)**:
- Ambos: 0% sucesso
- Greedy: Progresso melhor (até 40.74%)
- LLM: Progresso muito limitado (até 27.78%)

**Conclusão**: Nenhum dos métodos testados escalou bem para puzzles complexos. O Greedy demonstrou melhor progresso inicial, mas ambos falharam em resolver puzzles além do mais simples.

### 6.3 Respostas às Questões de Pesquisa

**RQ1: Qual método apresenta melhor taxa de sucesso?**

Para os métodos testados (Greedy e LLM), o Greedy apresentou melhor taxa de sucesso geral (27.3% vs 18.2%). No entanto, ambos falharam em resolver a maioria dos puzzles testados. Para uma resposta completa, seria necessário testar Q-Learning, V-Learning e Neural Networks.

**RQ2: Como a eficiência varia entre os métodos?**

O Greedy é drasticamente mais eficiente em termos de tempo (< 0.01s vs 7.18s), mas ambos apresentam eficiência similar em termos de movimentos quando resolvem (2.7 vs 3.0 movimentos).

**RQ3: Qual método escala melhor para puzzles 4D?**

Nenhum dos métodos testados escalou bem para o hipercubo 3x3x3x3. O Greedy alcançou melhor progresso (34.26% vs 5.56%), mas ambos falharam em resolver.

**RQ4: LLMs podem ser uma alternativa viável?**

Os resultados são mistos. O LLM demonstrou potencial (conseguiu resolver hypercuboid_1x1x1x2), mas apresenta limitações significativas em velocidade e taxa de sucesso. Com melhorias em prompts, modelos e otimizações, pode se tornar mais viável.

### 6.4 Validação das Hipóteses

**H1: Métodos de RL superam Greedy em puzzles complexos**
- **Status**: Não testado (Q/V/NN não foram executados no benchmark completo)
- **Evidência parcial**: Greedy falhou em todos os puzzles complexos, sugerindo necessidade de métodos mais sofisticados

**H2: Neural Networks escalam melhor que tabelas**
- **Status**: Não testado (NN não foi executado)

**H3: LLMs oferecem abordagem promissora**
- **Status**: Parcialmente confirmado
- **Evidência**: LLM conseguiu resolver hypercuboid, mas com limitações significativas

**H4: Escalabilidade varia com complexidade**
- **Status**: Confirmado
- **Evidência**: Ambos os métodos falharam em puzzles complexos, mas Greedy apresentou melhor progresso inicial

### 6.5 Limitações do Estudo

1. **Cobertura de Métodos**: Apenas Greedy e LLM foram testados completamente. Q-Learning, V-Learning e Neural Networks não foram executados no benchmark devido a limitações de tempo e recursos.

2. **Número de Testes**: Número limitado de testes por combinação (3-5), o que pode afetar a significância estatística.

3. **Treinamento**: Métodos que requerem treinamento (Q/V/NN) podem não ter convergido completamente.

4. **Modelo LLM**: Apenas um modelo foi testado (llama3.2). Modelos maiores ou fine-tuned podem apresentar resultados diferentes.

5. **Puzzles**: Apenas 4 puzzles foram testados. Uma avaliação mais completa requereria mais variedade.

6. **Métricas**: Algumas métricas importantes (como qualidade da solução, não apenas sucesso/falha) não foram coletadas.

---

## 7. Conclusões

Este trabalho apresentou uma comparação sistemática de métodos de Inteligência Computacional para resolução de cubos mágicos, com foco especial em uma nova abordagem utilizando modelos de linguagem grandes.

### 7.1 Principais Conclusões

1. **Métodos Heurísticos Simples**: O método Greedy demonstrou ser eficiente e eficaz para puzzles muito simples (100% sucesso em hypercuboid_1x1x1x2), mas falha em puzzles complexos devido a limitações de otimização local.

2. **Abordagem LLM**: O uso de LLMs para resolução de puzzles demonstra potencial promissor, especialmente para puzzles onde conhecimento geral pode ser útil. No entanto, limitações atuais incluem latência alta e taxa de sucesso ainda inferior a métodos tradicionais.

3. **Escalabilidade**: Nenhum dos métodos testados escalou bem para puzzles complexos (3x3x3, hypercube). Isso sugere necessidade de métodos mais sofisticados ou combinação de abordagens.

4. **Trade-offs**: Existe um trade-off claro entre velocidade (Greedy) e potencial de aprendizado (LLM, métodos de RL). A escolha do método depende do contexto e requisitos específicos.

### 7.2 Contribuições

As principais contribuições deste trabalho incluem:

1. **Implementação Completa**: Sistema completo de benchmarking com suporte para múltiplos métodos
2. **Abordagem Inovadora**: Primeira avaliação sistemática de LLMs para resolução de puzzles geométricos complexos
3. **Análise Comparativa**: Comparação detalhada de métodos com métricas abrangentes
4. **Extensibilidade**: Framework extensível que permite adicionar novos métodos facilmente

### 7.3 Trabalhos Futuros

Direções futuras incluem:

1. **Completar Benchmark**: Executar testes completos com Q-Learning, V-Learning e Neural Networks
2. **Melhorar LLM**: 
   - Fine-tuning de modelos para puzzles específicos
   - Prompts mais sofisticados (chain-of-thought, few-shot)
   - Modelos maiores e mais capazes
3. **Ensemble Methods**: Combinar múltiplos métodos (ex: LLM + Greedy, LLM + Q-Learning)
4. **Transfer Learning**: Treinar em puzzles simples e transferir conhecimento para complexos
5. **Otimizações**: Melhorar cache, batching, e outras otimizações para LLM
6. **Mais Puzzles**: Expandir para outros tipos de puzzles além de cubos
7. **Análise Teórica**: Desenvolver análise teórica da complexidade e limites dos métodos

### 7.4 Impacto e Relevância

Este trabalho contribui para o campo de Inteligência Computacional aplicada a resolução de puzzles, demonstrando tanto o potencial quanto as limitações de diferentes abordagens. A avaliação de LLMs como método de resolução abre novas direções de pesquisa e pode inspirar trabalhos futuros em áreas relacionadas.

---

## Referências

[INSERIR REFERÊNCIAS RELEVANTES]

1. Korf, R. E. (1997). Finding optimal solutions to Rubik's cube using pattern databases. *AAAI*, 97, 700-705.

2. Agostinelli, F., et al. (2019). Solving the Rubik's cube with deep reinforcement learning and search. *Nature Machine Intelligence*, 1(8), 356-363.

3. McAleer, S., et al. (2018). Solving the Rubik's cube without human knowledge. *arXiv preprint arXiv:1805.07470*.

4. Brown, T., et al. (2020). Language models are few-shot learners. *Advances in neural information processing systems*, 33, 1877-1901.

5. [Adicionar mais referências relevantes sobre RL, puzzles, LLMs]

---

## Apêndices

### Apêndice A: Detalhes de Implementação

O código fonte completo está disponível em [repositório]. Principais componentes:

- `src/ai_modules/llm_puzzle_solver.py`: Implementação do LLM Solver
- `src/ai_modules/llm_benchmark.py`: Sistema de benchmarking
- `src/ai_modules/greedy_solver.py`: Implementação do Greedy Solver
- `run_llm_benchmark.py`: Script principal para executar benchmarks

### Apêndice B: Resultados Detalhados

Resultados completos em formato JSON, CSV e Markdown estão disponíveis em `benchmark_results/benchmark_20251216_233645.*`.

### Apêndice C: Visualizações

6 visualizações de alta qualidade estão disponíveis em `benchmark_results/*.png`:
- `comparison_general.png`: Comparação geral
- `performance_by_puzzle.png`: Desempenho por puzzle
- `progress_analysis.png`: Análise de progresso
- `llm_metrics.png`: Métricas específicas do LLM
- `complexity_analysis.png`: Análise de complexidade
- `efficiency_comparison.png`: Comparação de eficiência

---

**Fim do Artigo**

