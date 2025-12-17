"""
Script principal para executar benchmark completo do LLM Puzzle Solver
Testa diferentes puzzles, compara métodos e gera relatórios
"""
import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from src.ai_modules.llm_benchmark import PuzzleBenchmark

def main():
    """Executa benchmark completo."""
    print("=" * 70)
    print("LLM Puzzle Solver - Benchmark Completo")
    print("=" * 70)
    print()
    
    # Verificar Ollama
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=3)
        if response.status_code != 200:
            print("[ERRO] Ollama nao esta respondendo corretamente")
            return
        print("[OK] Ollama conectado")
    except Exception as e:
        print(f"[ERRO] Ollama nao esta acessivel: {e}")
        print("Certifique-se que Ollama esta rodando: ollama serve")
        return
    
    # Criar benchmark
    benchmark = PuzzleBenchmark(output_dir="benchmark_results")
    
    # Configuração de testes - continuar até resolver ou timeout
    puzzles_to_test = [
        {
            "name": "cube_2x2x2",
            "scramble_length": 6,
            "max_moves": 10000,  # Continuar até resolver
            "max_time": 600.0,  # 10 minutos
            "num_tests": 3,
        },
        {
            "name": "cube_3x3x3",
            "scramble_length": 8,
            "max_moves": 10000,
            "max_time": 600.0,  # 10 minutos
            "num_tests": 2,  # Menos testes para 3x3x3 (mais lento)
        },
        {
            "name": "hypercuboid_1x1x1x2",
            "scramble_length": 5,
            "max_moves": 10000,
            "max_time": 300.0,  # 5 minutos
            "num_tests": 3,
        },
        {
            "name": "hypercube_3x3x3x3",
            "scramble_length": 4,
            "max_moves": 10000,
            "max_time": 600.0,  # 10 minutos (puzzle complexo)
            "num_tests": 3,
        },
    ]
    
    methods = ["llm", "greedy"]
    
    # Executar benchmarks
    for puzzle_config in puzzles_to_test:
        try:
            benchmark.run_benchmark(
                puzzle_name=puzzle_config["name"],
                methods=methods,
                num_tests=puzzle_config["num_tests"],
                scramble_length=puzzle_config["scramble_length"],
                max_moves=puzzle_config.get("max_moves", 10000),
                max_time=puzzle_config.get("max_time", 300.0),
                ollama_model="llama3.2",
            )
        except Exception as e:
            print(f"\n[ERRO] Falha ao testar {puzzle_config['name']}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Gerar relatórios
    print("\n" + "=" * 70)
    print("Gerando relatorios...")
    print("=" * 70)
    
    files = benchmark.generate_report(format="all")
    
    print("\n[OK] Relatorios gerados:")
    for format_type, filepath in files.items():
        print(f"  {format_type.upper()}: {filepath}")
    
    # Gerar visualizações
    if files.get("json"):
        print("\nGerando visualizacoes...")
        try:
            from src.ai_modules.visualize_benchmark import generate_all_visualizations
            generate_all_visualizations(files["json"], "benchmark_results")
        except ImportError:
            print("[AVISO] matplotlib nao disponivel, pulando visualizacoes")
        except Exception as e:
            print(f"[AVISO] Erro ao gerar visualizacoes: {e}")
    
    # Estatísticas resumidas
    print("\n" + "=" * 70)
    print("Estatisticas Resumidas")
    print("=" * 70)
    
    stats = benchmark.get_statistics()
    for method, method_stats in stats.items():
        print(f"\n{method.upper()}:")
        print(f"  Taxa de sucesso: {method_stats.get('success_rate', 0):.1f}%")
        if method_stats.get("avg_moves"):
            print(f"  Movimentos (media): {method_stats['avg_moves']:.1f}")
            print(f"  Tempo (media): {method_stats['avg_time']:.2f}s")
        if method == "llm" and method_stats.get("avg_llm_calls"):
            print(f"  Chamadas LLM (media): {method_stats['avg_llm_calls']:.1f}")
    
    print("\n" + "=" * 70)
    print("Benchmark concluido!")
    print("=" * 70)


if __name__ == "__main__":
    main()

