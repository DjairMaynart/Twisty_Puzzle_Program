"""
Benchmark rápido - testa apenas cubos pequenos para validação rápida
"""
import sys
import os

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from src.ai_modules.llm_benchmark import PuzzleBenchmark

def main():
    print("=" * 70)
    print("LLM Puzzle Solver - Benchmark Rapido")
    print("=" * 70)
    print()
    
    # Verificar Ollama
    try:
        import requests
        requests.get("http://localhost:11434/api/tags", timeout=3)
        print("[OK] Ollama conectado")
    except:
        print("[ERRO] Ollama nao esta acessivel")
        return
    
    benchmark = PuzzleBenchmark(output_dir="benchmark_results")
    
    # Teste rápido: apenas 2x2x2 - continuar até resolver
    print("\nExecutando benchmark rapido (2x2x2)...")
    benchmark.run_benchmark(
        puzzle_name="cube_2x2x2",
        methods=["llm", "greedy"],
        num_tests=2,
        scramble_length=6,
        max_moves=50000,  # Continuar até resolver (limite muito alto)
        max_time=600.0,  # 10 minutos para garantir que resolve
        ollama_model="llama3.2",
    )
    
    # Relatório
    files = benchmark.generate_report(format="markdown")
    if files:
        print(f"\n[OK] Relatorio gerado: {files.get('markdown', 'N/A')}")
    
    stats = benchmark.get_statistics()
    print("\nResumo:")
    for method, s in stats.items():
        print(f"  {method}: {s.get('success_rate', 0):.1f}% sucesso")

if __name__ == "__main__":
    main()

