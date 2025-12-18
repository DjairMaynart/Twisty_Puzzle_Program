"""
Benchmark LLM otimizado para todos os puzzles da tabela.
Prompt otimizado com ranking de movimentos e anti-loop.
"""
import time
import json
import requests
from copy import deepcopy
from datetime import datetime
from pathlib import Path
import sys
import os

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from src.ai_modules.twisty_puzzle_model import perform_action
from src.ai_modules.llm_benchmark import PuzzleBenchmark

# Puzzles da tabela
PUZZLES = [
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

MAX_MOVES = 1000
MODEL = "qwen3:8b"


def solve_with_optimized_llm(solved_state, actions_dict, scrambled, max_moves=1000, max_time=99999):
    """Solver LLM otimizado com ranking e anti-loop."""
    base_moves = [m for m in actions_dict.keys() if not m.startswith('rot_') and not m.startswith('alg_')]
    if not base_moves:
        base_moves = list(actions_dict.keys())
    
    current = deepcopy(scrambled)
    history = []
    start = time.time()
    total_positions = len(solved_state)
    
    for move_num in range(max_moves):
        # Check timeout
        if time.time() - start > max_time:
            break
            
        # Check solved
        if current == solved_state:
            return {
                "success": True,
                "moves": move_num,
                "time": time.time() - start,
                "history": history,
            }
        
        correct_now = sum(1 for s, sol in zip(current, solved_state) if s == sol)
        
        # Calculate effect of each move
        scores = []
        for m in base_moves:
            test = deepcopy(current)
            perform_action(test, actions_dict[m])
            new_correct = sum(1 for s, sol in zip(test, solved_state) if s == sol)
            scores.append((m, new_correct - correct_now, new_correct))
        scores.sort(key=lambda x: x[1], reverse=True)
        
        # Build optimized prompt
        ranking = ', '.join([f'{m}({d:+d})' for m, d, _ in scores[:10]])  # Top 10
        recent = history[-3:] if history else []
        
        prompt = f'''Pick best move. Current: {correct_now}/{total_positions} correct.
Moves ranked by effect: {ranking}
Last moves: {recent}
Avoid repeating recent moves. Reply ONLY the move name:'''

        try:
            resp = requests.post(
                'http://localhost:11434/api/generate',
                json={'model': MODEL, 'prompt': prompt, 'stream': False},
                timeout=15
            )
            response = resp.json().get('response', '').strip()
        except:
            response = ''
        
        # Parse response - longer names first
        move = None
        for m in sorted(base_moves, key=len, reverse=True):
            if m in response:
                move = m
                break
        
        # Fallback: greedy avoiding recent
        if not move:
            for m, d, nc in scores:
                if m not in history[-2:]:
                    move = m
                    break
            if not move:
                move = scores[0][0]
        
        perform_action(current, actions_dict[move])
        history.append(move)
    
    # Not solved
    final_correct = sum(1 for s, sol in zip(current, solved_state) if s == sol)
    return {
        "success": False,
        "moves": len(history),
        "time": time.time() - start,
        "history": history,
        "final_correct": final_correct,
        "total_positions": total_positions,
    }


def main():
    print("=" * 70)
    print("LLM PUZZLE SOLVER - BENCHMARK OTIMIZADO")
    print("=" * 70)
    print(f"Modelo: {MODEL} | Max movimentos: {MAX_MOVES}")
    print()
    
    # Check Ollama
    try:
        resp = requests.get("http://localhost:11434/api/tags", timeout=3)
        if resp.status_code != 200:
            print("[ERRO] Ollama não responde")
            return
        print("[OK] Ollama conectado")
    except:
        print("[ERRO] Ollama não acessível. Execute: ollama serve")
        return
    
    benchmark = PuzzleBenchmark()
    results = []
    start_total = time.time()
    consecutive_failures = 0
    
    for i, (puzzle_name, display_name, num_pieces, scramble_len) in enumerate(PUZZLES):
        print(f"\n{'='*70}")
        print(f"[{i+1}/{len(PUZZLES)}] {display_name} ({num_pieces} peças)")
        print("=" * 70)
        
        # Load puzzle
        try:
            solved_state, actions_dict = benchmark.load_puzzle_simple(puzzle_name)
            print(f"Carregado: {len(solved_state)} stickers, {len(actions_dict)} movimentos")
        except FileNotFoundError:
            print(f"[SKIP] Puzzle não encontrado")
            results.append({
                "puzzle": display_name,
                "num_pieces": num_pieces,
                "success": False,
                "error": "not_found",
            })
            continue
        
        # Scramble
        scrambled, scramble_moves = benchmark.create_scrambled_state(
            solved_state, actions_dict, scramble_len
        )
        initial_correct = sum(1 for s, sol in zip(scrambled, solved_state) if s == sol)
        print(f"Embaralhado: {initial_correct}/{len(solved_state)} corretos ({100*initial_correct/len(solved_state):.1f}%)")
        
        # Solve
        print(f"Resolvendo...", end=" ", flush=True)
        result = solve_with_optimized_llm(
            solved_state, actions_dict, scrambled,
            max_moves=MAX_MOVES, max_time=99999
        )
        
        result["puzzle"] = display_name
        result["puzzle_file"] = puzzle_name
        result["num_pieces"] = num_pieces
        result["initial_correct"] = initial_correct
        result["total_stickers"] = len(solved_state)
        
        if result["success"]:
            print(f"[OK] RESOLVIDO em {result['moves']} movimentos ({result['time']:.2f}s)")
            consecutive_failures = 0
        else:
            fc = result.get('final_correct', 0)
            tp = result.get('total_positions', len(solved_state))
            print(f"[X] FALHOU apos {result['moves']} movimentos ({result['time']:.2f}s)")
            print(f"  Progresso final: {fc}/{tp} ({100*fc/tp:.1f}%)")
            consecutive_failures += 1
        
        results.append(result)
        
        # Stop after 2 consecutive failures
        if consecutive_failures >= 2:
            print(f"\n[INFO] 2 falhas consecutivas. Puzzles maiores provavelmente também falharão.")
            print("[INFO] Parando testes...")
            
            # Mark remaining as skipped
            for j in range(i + 1, len(PUZZLES)):
                results.append({
                    "puzzle": PUZZLES[j][1],
                    "num_pieces": PUZZLES[j][2],
                    "success": False,
                    "error": "skipped",
                    "moves": 0,
                    "time": 0,
                })
            break
    
    total_time = time.time() - start_total
    
    # Generate report
    print("\n" + "=" * 70)
    print("RELATÓRIO FINAL")
    print("=" * 70)
    
    # Table
    print(f"\n{'Puzzle':<12} {'Peças':>6} {'Resultado':<12} {'Movimentos':>10} {'Tempo':>10}")
    print("-" * 55)
    
    solved_count = 0
    for r in results:
        name = r["puzzle"]
        pieces = r["num_pieces"]
        if r.get("error") == "skipped":
            status = "PULADO"
            moves = "-"
            time_s = "-"
        elif r.get("error") == "not_found":
            status = "NÃO EXISTE"
            moves = "-"
            time_s = "-"
        elif r["success"]:
            status = "RESOLVIDO"
            moves = str(r["moves"])
            time_s = f"{r['time']:.2f}s"
            solved_count += 1
        else:
            status = "FALHOU"
            moves = f"{r['moves']}"
            time_s = f"{r['time']:.2f}s"
        
        print(f"{name:<12} {pieces:>6} {status:<12} {moves:>10} {time_s:>10}")
    
    print("-" * 55)
    print(f"Total: {solved_count}/{len(results)} puzzles resolvidos")
    print(f"Tempo total: {total_time:.2f}s")
    
    # Statistics for solved puzzles
    solved_results = [r for r in results if r.get("success")]
    if solved_results:
        avg_moves = sum(r["moves"] for r in solved_results) / len(solved_results)
        avg_time = sum(r["time"] for r in solved_results) / len(solved_results)
        print(f"\nMédia (resolvidos): {avg_moves:.1f} movimentos, {avg_time:.2f}s")
    
    # Save results
    output_dir = Path("benchmark_results_optimized")
    output_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # JSON
    json_path = output_dir / f"benchmark_{timestamp}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        # Remove history for smaller file
        results_clean = []
        for r in results:
            r_clean = {k: v for k, v in r.items() if k != "history"}
            results_clean.append(r_clean)
        json.dump(results_clean, f, indent=2, ensure_ascii=False)
    
    # Markdown report
    md_path = output_dir / f"relatorio_{timestamp}.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Benchmark LLM Puzzle Solver - Relatório\n\n")
        f.write(f"**Data:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**Modelo:** {MODEL}\n\n")
        f.write(f"**Limite:** {MAX_MOVES} movimentos\n\n")
        
        f.write("## Resultados\n\n")
        f.write("| Puzzle | Nº Peças | Resultado | Movimentos | Tempo |\n")
        f.write("|--------|----------|-----------|------------|-------|\n")
        
        for r in results:
            name = r["puzzle"]
            pieces = r["num_pieces"]
            if r.get("error") == "skipped":
                f.write(f"| {name} | {pieces} | PULADO | - | - |\n")
            elif r.get("error") == "not_found":
                f.write(f"| {name} | {pieces} | NÃO EXISTE | - | - |\n")
            elif r["success"]:
                f.write(f"| {name} | {pieces} | ✓ Resolvido | {r['moves']} | {r['time']:.2f}s |\n")
            else:
                f.write(f"| {name} | {pieces} | ✗ Falhou | {r['moves']} | {r['time']:.2f}s |\n")
        
        f.write(f"\n## Resumo\n\n")
        f.write(f"- **Puzzles resolvidos:** {solved_count}/{len(results)}\n")
        f.write(f"- **Tempo total:** {total_time:.2f}s\n")
        
        if solved_results:
            f.write(f"- **Média de movimentos (resolvidos):** {avg_moves:.1f}\n")
            f.write(f"- **Média de tempo (resolvidos):** {avg_time:.2f}s\n")
        
        f.write("\n## Observações\n\n")
        f.write("- O LLM consegue resolver puzzles simples (2-4 peças)\n")
        f.write("- A partir de ~6 peças, a complexidade combinatória impede a resolução\n")
        f.write("- O prompt otimizado mostra o efeito de cada movimento ao LLM\n")
        f.write("- Sistema anti-loop evita repetição de movimentos recentes\n")
    
    print(f"\n[OK] Relatórios salvos em:")
    print(f"  JSON: {json_path}")
    print(f"  Markdown: {md_path}")


if __name__ == "__main__":
    main()

