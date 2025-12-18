"""
Script para benchmark de LLM em todos os puzzles da tabela.
Limite de 1000 movimentos, coleta métricas de tempo e movimentos.
"""
import sys
import os
import time
import json
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from src.ai_modules.llm_benchmark import PuzzleBenchmark

# Lista de puzzles da tabela (nome no sistema)
PUZZLES_TABLE = [
    # (nome_arquivo, nome_display, num_pecas, scramble_length)
    ("hypercuboid_1x1x1x2", "1x1x1x2", 2, 3),
    ("hypercuboid_1x1x1x3", "1x1x1x3", 3, 4),
    ("hypercuboid_1x1x2x2", "1x1x2x2", 4, 4),
    ("hypercuboid_1x1x2x3", "1x1x2x3", 6, 5),
    ("hypercuboid_1x1x3x3", "1x1x3x3", 9, 6),
    ("hypercuboid_1x2x2x2", "1x2x2x2", 8, 6),
    ("hypercuboid_1x2x2x3", "1x2x2x3", 12, 7),
    ("hypercuboid_1x2x3x3", "1x2x3x3", 18, 8),
    ("hypercuboid_1x3x3x3", "1x3x3x3", 27, 8),
    ("cube_2x2x2", "2x2x2x2", 16, 6),
    ("hypercuboid_2x2x2x3", "2x2x2x3", 24, 7),
    ("hypercuboid_2x2x3x3", "2x2x3x3", 36, 8),
    ("hypercuboid_2x3x3x3", "2x3x3x3", 54, 8),
    ("cube_3x3x3", "3x3x3x3", 80, 8),
]

def check_puzzle_exists(benchmark, puzzle_name):
    """Verifica se o puzzle existe."""
    try:
        benchmark.load_puzzle_simple(puzzle_name)
        return True
    except FileNotFoundError:
        return False

def main():
    print("=" * 70)
    print("LLM Puzzle Solver - Benchmark TODOS os Puzzles")
    print("Limite: 1000 movimentos | Métrica: Tempo + Movimentos")
    print("=" * 70)
    print()
    
    # Verificar Ollama
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=3)
        if response.status_code != 200:
            print("[ERRO] Ollama não está respondendo")
            return
        print("[OK] Ollama conectado")
        
        # Mostrar modelos disponíveis
        models = response.json().get("models", [])
        print(f"[INFO] Modelos disponíveis: {[m['name'] for m in models]}")
    except Exception as e:
        print(f"[ERRO] Ollama não acessível: {e}")
        print("Execute: ollama serve")
        return
    
    # Criar benchmark
    output_dir = "benchmark_results_all_puzzles"
    os.makedirs(output_dir, exist_ok=True)
    benchmark = PuzzleBenchmark(output_dir=output_dir)
    
    # Verificar quais puzzles existem
    print("\n[INFO] Verificando puzzles disponíveis...")
    available_puzzles = []
    for puzzle_name, display_name, num_pieces, scramble_len in PUZZLES_TABLE:
        if check_puzzle_exists(benchmark, puzzle_name):
            available_puzzles.append((puzzle_name, display_name, num_pieces, scramble_len))
            print(f"  [OK] {display_name} ({puzzle_name})")
        else:
            print(f"  [FALTA] {display_name} ({puzzle_name})")
    
    if not available_puzzles:
        print("\n[ERRO] Nenhum puzzle encontrado!")
        return
    
    print(f"\n[INFO] {len(available_puzzles)} puzzles disponíveis para teste")
    
    # Configurações
    MAX_MOVES = 1000
    MAX_TIME = 120.0  # 2 minutos por teste
    NUM_TESTS = 1  # 1 teste por puzzle para ser rápido
    
    all_results = []
    start_total = time.time()
    
    # Testar cada puzzle
    for i, (puzzle_name, display_name, num_pieces, scramble_len) in enumerate(available_puzzles):
        print(f"\n{'='*70}")
        print(f"[{i+1}/{len(available_puzzles)}] Testando: {display_name}")
        print(f"{'='*70}")
        
        try:
            results = benchmark.run_benchmark(
                puzzle_name=puzzle_name,
                methods=["llm"],  # Só LLM
                num_tests=NUM_TESTS,
                scramble_length=scramble_len,
                max_moves=MAX_MOVES,
                max_time=MAX_TIME,
                ollama_model="llama3.2",
            )
            
            # Processar resultado
            for r in results:
                r["display_name"] = display_name
                r["num_pieces"] = num_pieces
                all_results.append(r)
                
                # Mostrar resultado
                if r.get("success"):
                    print(f"  [RESOLVIDO] {r['moves']} movimentos em {r['time']:.2f}s")
                else:
                    print(f"  [FALHOU] Após {r['moves']} movimentos ({r['time']:.2f}s)")
                    print(f"    Progresso: {r.get('final_correct', 0)}/{r.get('total_positions', 0)}")
                    
                    # Se não conseguiu, parar de testar puzzles maiores
                    if r['moves'] >= MAX_MOVES:
                        print(f"\n[INFO] Limite de {MAX_MOVES} movimentos atingido.")
                        print("[INFO] Puzzles maiores provavelmente também não serão resolvidos.")
                        print("[INFO] Parando testes...")
                        break
            
            # Verificar se deve parar
            if results and not results[-1].get("success") and results[-1].get("moves", 0) >= MAX_MOVES:
                break
                
        except Exception as e:
            print(f"  [ERRO] {e}")
            all_results.append({
                "puzzle": puzzle_name,
                "display_name": display_name,
                "num_pieces": num_pieces,
                "method": "llm",
                "success": False,
                "error": str(e),
                "moves": 0,
                "time": 0,
            })
    
    total_time = time.time() - start_total
    
    # Salvar resultados
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # JSON
    json_path = os.path.join(output_dir, f"all_puzzles_{timestamp}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
    
    # Relatório resumido
    print("\n" + "=" * 70)
    print("RESULTADOS FINAIS")
    print("=" * 70)
    print(f"\n{'Puzzle':<15} {'Peças':<8} {'Resultado':<12} {'Movimentos':<12} {'Tempo':<10}")
    print("-" * 60)
    
    for r in all_results:
        name = r.get("display_name", r.get("puzzle", "?"))
        pieces = r.get("num_pieces", "?")
        status = "RESOLVIDO" if r.get("success") else "FALHOU"
        moves = r.get("moves", 0)
        time_s = r.get("time", 0)
        print(f"{name:<15} {pieces:<8} {status:<12} {moves:<12} {time_s:.2f}s")
    
    print("-" * 60)
    print(f"Tempo total: {total_time:.2f}s")
    print(f"Resultados salvos em: {json_path}")
    
    # Estatísticas
    successful = [r for r in all_results if r.get("success")]
    print(f"\nResumo: {len(successful)}/{len(all_results)} puzzles resolvidos")
    
    if successful:
        avg_moves = sum(r["moves"] for r in successful) / len(successful)
        avg_time = sum(r["time"] for r in successful) / len(successful)
        print(f"Média (resolvidos): {avg_moves:.1f} movimentos, {avg_time:.2f}s")


if __name__ == "__main__":
    main()

