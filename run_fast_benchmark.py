"""
Benchmark LLM RAPIDO - usando lib ollama diretamente.
~0.2s por movimento apos warmup.
"""
import time
import json
import ollama
from copy import deepcopy
from datetime import datetime
from pathlib import Path
import sys
import os

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from src.ai_modules.twisty_puzzle_model import perform_action
from src.ai_modules.llm_benchmark import PuzzleBenchmark

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
MODEL = "llama3.2"


def warmup_model():
    """Carrega modelo na GPU."""
    print(f"[INFO] Carregando {MODEL} na GPU...", end=" ", flush=True)
    ollama.generate(model=MODEL, prompt="test", options={'num_predict': 1})
    print("OK")


def solve_fast(solved_state, actions_dict, scrambled, max_moves=1000):
    """Solver rapido usando lib ollama."""
    base_moves = [m for m in actions_dict.keys() if not m.startswith('rot_') and not m.startswith('alg_')]
    if not base_moves:
        base_moves = list(actions_dict.keys())
    
    current = deepcopy(scrambled)
    history = []
    start = time.time()
    total_pos = len(solved_state)
    
    for move_num in range(max_moves):
        if current == solved_state:
            return {"success": True, "moves": move_num, "time": time.time() - start}
        
        correct_now = sum(1 for s, sol in zip(current, solved_state) if s == sol)
        
        # Calcular efeitos
        scores = []
        for m in base_moves:
            test = deepcopy(current)
            perform_action(test, actions_dict[m])
            nc = sum(1 for s, sol in zip(test, solved_state) if s == sol)
            scores.append((m, nc - correct_now, nc))
        scores.sort(key=lambda x: x[1], reverse=True)
        
        # Prompt minimo
        ranking = ','.join([f'{m}({d:+d})' for m, d, _ in scores[:8]])
        recent = ','.join(history[-3:]) if history else 'none'
        
        prompt = f'{correct_now}/{total_pos}ok. Moves:{ranking}. Last:{recent}. Best move:'
        
        try:
            resp = ollama.generate(
                model=MODEL, 
                prompt=prompt, 
                options={'num_predict': 15, 'temperature': 0.1}
            )
            response = resp['response'].strip()
        except:
            response = ''
        
        # Parse - maior primeiro
        move = None
        for m in sorted(base_moves, key=len, reverse=True):
            if m in response:
                move = m
                break
        
        # Fallback greedy
        if not move:
            for m, d, nc in scores:
                if m not in history[-2:]:
                    move = m
                    break
            if not move:
                move = scores[0][0]
        
        perform_action(current, actions_dict[move])
        history.append(move)
    
    final_correct = sum(1 for s, sol in zip(current, solved_state) if s == sol)
    return {
        "success": False, 
        "moves": len(history), 
        "time": time.time() - start,
        "final_correct": final_correct,
        "total_positions": total_pos,
    }


def main():
    print("=" * 60)
    print("LLM PUZZLE SOLVER - BENCHMARK RAPIDO")
    print("=" * 60)
    print(f"Modelo: {MODEL} | Max: {MAX_MOVES} movimentos")
    print()
    
    warmup_model()
    
    benchmark = PuzzleBenchmark()
    results = []
    start_total = time.time()
    consecutive_failures = 0
    
    for i, (puzzle_name, display_name, num_pieces, scramble_len) in enumerate(PUZZLES):
        print(f"\n[{i+1}/{len(PUZZLES)}] {display_name} ({num_pieces} pecas)")
        
        try:
            solved_state, actions_dict = benchmark.load_puzzle_simple(puzzle_name)
        except FileNotFoundError:
            print(f"  [SKIP] Nao encontrado")
            results.append({"puzzle": display_name, "num_pieces": num_pieces, "success": False, "error": "not_found"})
            continue
        
        scrambled, _ = benchmark.create_scrambled_state(solved_state, actions_dict, scramble_len)
        initial_correct = sum(1 for s, sol in zip(scrambled, solved_state) if s == sol)
        print(f"  {len(solved_state)} stickers, {initial_correct}/{len(solved_state)} corretos inicial")
        
        result = solve_fast(solved_state, actions_dict, scrambled, MAX_MOVES)
        result["puzzle"] = display_name
        result["num_pieces"] = num_pieces
        result["initial_correct"] = initial_correct
        result["total_stickers"] = len(solved_state)
        
        if result["success"]:
            print(f"  [OK] RESOLVIDO: {result['moves']} mov, {result['time']:.1f}s")
            consecutive_failures = 0
        else:
            fc = result.get('final_correct', 0)
            tp = result.get('total_positions', len(solved_state))
            print(f"  [X] FALHOU: {result['moves']} mov, {result['time']:.1f}s ({fc}/{tp})")
            consecutive_failures += 1
        
        results.append(result)
        
        if consecutive_failures >= 2:
            print(f"\n[INFO] 2 falhas consecutivas, parando...")
            for j in range(i + 1, len(PUZZLES)):
                results.append({"puzzle": PUZZLES[j][1], "num_pieces": PUZZLES[j][2], "success": False, "error": "skipped", "moves": 0, "time": 0})
            break
    
    total_time = time.time() - start_total
    
    # Relatorio
    print("\n" + "=" * 60)
    print("RESULTADOS")
    print("=" * 60)
    print(f"{'Puzzle':<12} {'Pecas':>6} {'Status':<10} {'Mov':>6} {'Tempo':>8}")
    print("-" * 50)
    
    solved_count = 0
    for r in results:
        name = r["puzzle"]
        pieces = r["num_pieces"]
        if r.get("error"):
            status, moves, time_s = r["error"].upper(), "-", "-"
        elif r["success"]:
            status, moves, time_s = "OK", str(r["moves"]), f"{r['time']:.1f}s"
            solved_count += 1
        else:
            status, moves, time_s = "FALHOU", str(r["moves"]), f"{r['time']:.1f}s"
        print(f"{name:<12} {pieces:>6} {status:<10} {moves:>6} {time_s:>8}")
    
    print("-" * 50)
    print(f"Resolvidos: {solved_count}/{len(results)} | Tempo total: {total_time:.1f}s")
    
    # Salvar
    output_dir = Path("benchmark_results_fast")
    output_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    json_path = output_dir / f"benchmark_{timestamp}.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    
    md_path = output_dir / f"relatorio_{timestamp}.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# Benchmark LLM - {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
        f.write(f"**Modelo:** {MODEL} | **Limite:** {MAX_MOVES} movimentos\n\n")
        f.write("| Puzzle | Pecas | Status | Movimentos | Tempo |\n")
        f.write("|--------|-------|--------|------------|-------|\n")
        for r in results:
            name = r["puzzle"]
            pieces = r["num_pieces"]
            if r.get("error"):
                f.write(f"| {name} | {pieces} | {r['error']} | - | - |\n")
            elif r["success"]:
                f.write(f"| {name} | {pieces} | OK | {r['moves']} | {r['time']:.1f}s |\n")
            else:
                f.write(f"| {name} | {pieces} | FALHOU | {r['moves']} | {r['time']:.1f}s |\n")
        f.write(f"\n**Total:** {solved_count}/{len(results)} resolvidos em {total_time:.1f}s\n")
    
    print(f"\nSalvo em: {json_path}")


if __name__ == "__main__":
    main()

