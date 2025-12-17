"""
Visualizações para resultados de benchmark
"""
import json
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from typing import Dict, List
import statistics


def load_benchmark_results(json_path: str) -> List[Dict]:
    """Carrega resultados de benchmark de arquivo JSON."""
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def plot_comparison(results: List[Dict], output_path: str = "benchmark_comparison.png"):
    """Gera gráfico comparativo entre métodos."""
    # Agrupar por método
    by_method = {}
    for result in results:
        method = result.get("method", "unknown")
        if method not in by_method:
            by_method[method] = {"moves": [], "times": [], "success": []}
        
        if result.get("success"):
            by_method[method]["moves"].append(result.get("moves", 0))
            by_method[method]["times"].append(result.get("time", 0))
        by_method[method]["success"].append(result.get("success", False))
    
    # Criar figura com subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Benchmark Comparison - LLM vs Other Methods", fontsize=16)
    
    # 1. Taxa de sucesso
    ax = axes[0, 0]
    methods = list(by_method.keys())
    success_rates = [
        sum(by_method[m]["success"]) / len(by_method[m]["success"]) * 100 
        if by_method[m]["success"] else 0
        for m in methods
    ]
    ax.bar(methods, success_rates, color=['#2ecc71', '#3498db', '#e74c3c', '#f39c12'])
    ax.set_ylabel("Taxa de Sucesso (%)")
    ax.set_title("Taxa de Sucesso por Método")
    ax.set_ylim(0, 100)
    
    # 2. Movimentos médios
    ax = axes[0, 1]
    avg_moves = [
        statistics.mean(by_method[m]["moves"]) if by_method[m]["moves"] else 0
        for m in methods
    ]
    ax.bar(methods, avg_moves, color=['#2ecc71', '#3498db', '#e74c3c', '#f39c12'])
    ax.set_ylabel("Movimentos (média)")
    ax.set_title("Número de Movimentos por Método")
    
    # 3. Tempo médio
    ax = axes[1, 0]
    avg_times = [
        statistics.mean(by_method[m]["times"]) if by_method[m]["times"] else 0
        for m in methods
    ]
    ax.bar(methods, avg_times, color=['#2ecc71', '#3498db', '#e74c3c', '#f39c12'])
    ax.set_ylabel("Tempo (segundos)")
    ax.set_title("Tempo de Resolução por Método")
    
    # 4. Movimentos vs Tempo (scatter)
    ax = axes[1, 1]
    for method in methods:
        moves = by_method[method]["moves"]
        times = by_method[method]["times"]
        if moves and times:
            ax.scatter(moves, times, label=method, alpha=0.6, s=50)
    ax.set_xlabel("Movimentos")
    ax.set_ylabel("Tempo (s)")
    ax.set_title("Movimentos vs Tempo")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Gráfico salvo: {output_path}")
    return output_path


def plot_by_puzzle(results: List[Dict], output_path: str = "benchmark_by_puzzle.png"):
    """Gera gráfico comparativo por puzzle."""
    # Agrupar por puzzle e método
    by_puzzle = {}
    for result in results:
        puzzle = result.get("puzzle", "unknown")
        method = result.get("method", "unknown")
        
        if puzzle not in by_puzzle:
            by_puzzle[puzzle] = {}
        if method not in by_puzzle[puzzle]:
            by_puzzle[puzzle][method] = {"moves": [], "times": [], "success": []}
        
        if result.get("success"):
            by_puzzle[puzzle][method]["moves"].append(result.get("moves", 0))
            by_puzzle[puzzle][method]["times"].append(result.get("time", 0))
        by_puzzle[puzzle][method]["success"].append(result.get("success", False))
    
    puzzles = list(by_puzzle.keys())
    methods = set()
    for puzzle_data in by_puzzle.values():
        methods.update(puzzle_data.keys())
    methods = sorted(list(methods))
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Performance por Puzzle", fontsize=16)
    
    # Movimentos por puzzle
    ax = axes[0]
    x = range(len(puzzles))
    width = 0.35
    
    for i, method in enumerate(methods):
        moves_data = [
            statistics.mean(by_puzzle[p][method]["moves"]) 
            if method in by_puzzle[p] and by_puzzle[p][method]["moves"] 
            else 0
            for p in puzzles
        ]
        ax.bar([xi + i*width for xi in x], moves_data, width, label=method, alpha=0.8)
    
    ax.set_xlabel("Puzzle")
    ax.set_ylabel("Movimentos (média)")
    ax.set_title("Movimentos por Puzzle")
    ax.set_xticks([xi + width*(len(methods)-1)/2 for xi in x])
    ax.set_xticklabels(puzzles, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Tempo por puzzle
    ax = axes[1]
    for i, method in enumerate(methods):
        times_data = [
            statistics.mean(by_puzzle[p][method]["times"]) 
            if method in by_puzzle[p] and by_puzzle[p][method]["times"] 
            else 0
            for p in puzzles
        ]
        ax.bar([xi + i*width for xi in x], times_data, width, label=method, alpha=0.8)
    
    ax.set_xlabel("Puzzle")
    ax.set_ylabel("Tempo (s)")
    ax.set_title("Tempo por Puzzle")
    ax.set_xticks([xi + width*(len(methods)-1)/2 for xi in x])
    ax.set_xticklabels(puzzles, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Gráfico salvo: {output_path}")
    return output_path


def plot_llm_metrics(results: List[Dict], output_path: str = "llm_metrics.png"):
    """Gera gráficos específicos para métricas do LLM."""
    llm_results = [r for r in results if r.get("method") == "llm"]
    
    if not llm_results:
        print("Nenhum resultado LLM encontrado para visualizar")
        return None
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("LLM Solver - Métricas Detalhadas", fontsize=16)
    
    # 1. Chamadas LLM vs Movimentos
    ax = axes[0, 0]
    moves = [r.get("moves", 0) for r in llm_results if r.get("success")]
    llm_calls = [r.get("llm_calls", 0) for r in llm_results if r.get("success")]
    if moves and llm_calls:
        ax.scatter(moves, llm_calls, alpha=0.6)
        ax.set_xlabel("Movimentos")
        ax.set_ylabel("Chamadas ao LLM")
        ax.set_title("Chamadas LLM vs Movimentos")
        ax.grid(True, alpha=0.3)
    
    # 2. Eficiência do cache
    ax = axes[0, 1]
    cache_hits = [r.get("cache_hits", 0) for r in llm_results if r.get("success")]
    total_calls = [r.get("llm_calls", 0) + r.get("cache_hits", 0) for r in llm_results if r.get("success")]
    if cache_hits and total_calls:
        cache_rates = [h/t if t > 0 else 0 for h, t in zip(cache_hits, total_calls)]
        ax.hist(cache_rates, bins=20, alpha=0.7, color='green')
        ax.set_xlabel("Taxa de Cache Hit")
        ax.set_ylabel("Frequência")
        ax.set_title("Distribuição de Cache Hits")
        ax.grid(True, alpha=0.3, axis='y')
    
    # 3. Tempo por chamada LLM
    ax = axes[1, 0]
    times = [r.get("time", 0) for r in llm_results if r.get("success")]
    if times and llm_calls:
        time_per_call = [t/max(c, 1) for t, c in zip(times, llm_calls)]
        ax.hist(time_per_call, bins=20, alpha=0.7, color='blue')
        ax.set_xlabel("Tempo por Chamada LLM (s)")
        ax.set_ylabel("Frequência")
        ax.set_title("Distribuição de Tempo por Chamada")
        ax.grid(True, alpha=0.3, axis='y')
    
    # 4. Movimentos vs Tempo (com sucesso)
    ax = axes[1, 1]
    if moves and times:
        ax.scatter(moves, times, alpha=0.6, color='red')
        ax.set_xlabel("Movimentos")
        ax.set_ylabel("Tempo (s)")
        ax.set_title("Eficiência: Movimentos vs Tempo")
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Gráfico salvo: {output_path}")
    return output_path


def generate_all_visualizations(json_path: str, output_dir: str = "benchmark_results"):
    """Gera todas as visualizações a partir de um arquivo JSON."""
    results = load_benchmark_results(json_path)
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    print("Gerando visualizacoes...")
    
    plot_comparison(results, str(output_path / "comparison.png"))
    plot_by_puzzle(results, str(output_path / "by_puzzle.png"))
    plot_llm_metrics(results, str(output_path / "llm_metrics.png"))
    
    print("\nTodas as visualizacoes geradas!")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        json_path = sys.argv[1]
        generate_all_visualizations(json_path)
    else:
        # Procurar o JSON mais recente
        import glob
        json_files = glob.glob("benchmark_results/benchmark_*.json")
        if json_files:
            latest = max(json_files, key=lambda x: Path(x).stat().st_mtime)
            print(f"Usando arquivo mais recente: {latest}")
            generate_all_visualizations(latest)
        else:
            print("Nenhum arquivo JSON encontrado em benchmark_results/")

