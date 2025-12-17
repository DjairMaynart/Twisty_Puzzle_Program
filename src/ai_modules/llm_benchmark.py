"""
Sistema de benchmarking e métricas para LLM Puzzle Solver
Compara desempenho entre diferentes métodos de resolução
"""
import time
import json
import csv
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
try:
    import statistics
except ImportError:
    # Fallback para Python < 3.4
    def mean(data):
        return sum(data) / len(data) if data else 0
    def median(data):
        sorted_data = sorted(data)
        n = len(sorted_data)
        if n == 0:
            return 0
        if n % 2 == 0:
            return (sorted_data[n//2 - 1] + sorted_data[n//2]) / 2
        return sorted_data[n//2]
    class statistics:
        mean = staticmethod(mean)
        median = staticmethod(median)

from .llm_puzzle_solver import LLM_Puzzle_Solver
from .greedy_solver import Greedy_Puzzle_Solver
from .twisty_puzzle_model import perform_action, scramble


class PuzzleBenchmark:
    """Sistema de benchmarking para comparar diferentes métodos de resolução."""
    
    def __init__(self, output_dir: str = "benchmark_results"):
        """
        Inicializa o sistema de benchmarking.
        
        Args:
            output_dir: Diretório para salvar resultados
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.results = []
        
    def load_puzzle_simple(self, puzzle_name: str):
        """Load puzzle from XML without vpython dependency."""
        import xml.etree.ElementTree as ET
        import os
        
        project_root = Path(__file__).parent.parent.parent
        puzzle_path = project_root / "src" / "puzzles" / puzzle_name / "puzzle_definition.xml"
        
        if not puzzle_path.exists():
            raise FileNotFoundError(f"Puzzle não encontrado: {puzzle_path}")
        
        tree = ET.parse(puzzle_path)
        root = tree.getroot()
        
        # Get points
        points = []
        for point_elem in root.findall("points/point"):
            coords = point_elem.find("coords")
            color = point_elem.find("color")
            
            point_data = {
                "coords": (
                    float(coords.get("x")),
                    float(coords.get("y")),
                    float(coords.get("z"))
                ),
                "color": (
                    float(color.get("r")),
                    float(color.get("g")),
                    float(color.get("b"))
                )
            }
            points.append(point_data)
        
        # Get moves - parsear diretamente do XML
        actions_dict = {}
        for move_elem in root.findall("moves/move"):
            move_name = move_elem.get("name")
            cycles = []
            for cycle_elem in move_elem.findall("cycle"):
                # Formato: "0, 16, 11, 20"
                cycle_text = cycle_elem.text.strip()
                if cycle_text:
                    cycle = [int(i.strip()) for i in cycle_text.split(",")]
                    if cycle:
                        cycles.append(cycle)
            if cycles:
                actions_dict[move_name] = cycles
        
        # Convert to solved state
        colors = list(set(p["color"] for p in points))
        solved_state = [colors.index(p["color"]) for p in points]
        
        return solved_state, actions_dict
    
    def create_scrambled_state(
        self, 
        solved_state: List[int], 
        actions_dict: Dict,
        scramble_length: int = 10,
        min_correct: float = 0.3
    ) -> Tuple[List[int], List[str]]:
        """
        Cria um estado embaralhado garantindo que não está resolvido.
        Usa estratégia inteligente para evitar movimentos que se cancelam.
        
        Args:
            solved_state: Estado resolvido
            actions_dict: Dicionário de ações
            scramble_length: Número de movimentos para embaralhar
            min_correct: Porcentagem mínima de posições corretas (para garantir dificuldade)
            
        Returns:
            Tupla (estado embaralhado, sequência de movimentos)
        """
        import random
        from copy import deepcopy
        
        base_moves = [m for m in actions_dict.keys() 
                     if not m.startswith("rot_") and not m.startswith("alg_")]
        
        if not base_moves:
            base_moves = list(actions_dict.keys())
        
        scrambled = deepcopy(solved_state)
        scramble_moves = []
        visited_states = {tuple(solved_state)}  # Rastrear estados visitados
        
        # Aplicar movimentos garantindo que cada um mude o estado
        attempts = 0
        max_attempts = scramble_length * 3
        
        while len(scramble_moves) < scramble_length and attempts < max_attempts:
            attempts += 1
            
            # Escolher movimento aleatório
            move = random.choice(base_moves)
            
            # Aplicar movimento
            new_state = deepcopy(scrambled)
            perform_action(new_state, actions_dict[move])
            new_state_tuple = tuple(new_state)
            
            # Verificar se o estado mudou
            if new_state_tuple not in visited_states:
                scrambled = new_state
                scramble_moves.append(move)
                visited_states.add(new_state_tuple)
        
        # Verificar se está realmente embaralhado
        correct = sum(1 for s, sol in zip(scrambled, solved_state) if s == sol)
        correct_pct = correct / len(solved_state) if solved_state else 0
        
        # Se ainda estiver muito resolvido, aplicar mais movimentos únicos
        if correct_pct > (1.0 - min_correct):
            for _ in range(scramble_length):
                move = random.choice(base_moves)
                new_state = deepcopy(scrambled)
                perform_action(new_state, actions_dict[move])
                new_state_tuple = tuple(new_state)
                
                if new_state_tuple not in visited_states:
                    scrambled = new_state
                    scramble_moves.append(move)
                    visited_states.add(new_state_tuple)
                    
                    correct = sum(1 for s, sol in zip(scrambled, solved_state) if s == sol)
                    correct_pct = correct / len(solved_state) if solved_state else 0
                    
                    if correct_pct < (1.0 - min_correct):
                        break
        
        return scrambled, scramble_moves
    
    def solve_with_method(
        self,
        start_state: List[int],
        actions_dict: Dict,
        solved_state: List[int],
        method: str,
        max_moves: int = 10000,  # Limite muito alto - continuar até resolver
        max_time: float = 300.0,  # Timeout de 5 minutos por padrão
        **kwargs
    ) -> Dict:
        """
        Resolve um puzzle usando um método específico.
        Continua até resolver ou atingir timeout.
        
        Args:
            start_state: Estado inicial embaralhado
            actions_dict: Dicionário de ações
            solved_state: Estado resolvido
            method: Método a usar ('llm', 'greedy', 'q', 'v', 'nn')
            max_moves: Número máximo de movimentos (padrão muito alto)
            max_time: Tempo máximo em segundos (padrão 5 minutos)
            **kwargs: Parâmetros adicionais para o método
            
        Returns:
            Dicionário com métricas da resolução
        """
        start_time = time.time()
        move_sequence = []
        success = False
        states_visited = 0
        llm_calls = 0
        cache_hits = 0
        progress_history = []  # Histórico de progresso
        time_history = []  # Histórico de tempo
        
        current_state = start_state.copy()
        initial_correct = sum(1 for s, sol in zip(start_state, solved_state) if s == sol)
        
        if method == "llm":
            solver = LLM_Puzzle_Solver(
                ACTIONS_DICT=actions_dict,
                SOLVED_STATE=solved_state,
                ollama_model=kwargs.get("ollama_model", "llama3.2"),
                cache_responses=True,
                puzzle_name=kwargs.get("puzzle_name", "puzzle"),
            )
            
            # Resetar contadores
            solver.total_queries = 0
            solver.cache_hits = 0
            
            move_num = 0
            last_progress_time = start_time
            seen_states = set()  # Detectar loops
            state_history = []  # Últimos estados para detectar ciclos
            max_recent_states = 20  # Verificar últimos 20 estados
            
            while move_num < max_moves:
                # Verificar timeout
                elapsed_time = time.time() - start_time
                if elapsed_time > max_time:
                    break
                
                # Verificar se resolveu
                if current_state == solved_state:
                    success = True
                    break
                
                # Detectar loops - verificar se estado já foi visto recentemente
                state_tuple = tuple(current_state)
                if state_tuple in seen_states:
                    # Estado repetido - pode estar em loop
                    if len(state_history) >= max_recent_states:
                        # Verificar se é um ciclo
                        if state_tuple in state_history[-max_recent_states:]:
                            # Loop detectado - tentar movimento diferente
                            pass
                
                seen_states.add(state_tuple)
                state_history.append(state_tuple)
                if len(state_history) > max_recent_states:
                    state_history.pop(0)
                
                # Registrar progresso a cada 5 movimentos
                if move_num % 5 == 0:
                    current_correct = sum(1 for s, sol in zip(current_state, solved_state) if s == sol)
                    progress_history.append({
                        "move": move_num,
                        "correct": current_correct,
                        "time": elapsed_time
                    })
                    time_history.append(elapsed_time)
                
                # Escolher e aplicar movimento
                move = solver.choose_action(current_state)
                move_sequence.append(move)
                perform_action(current_state, actions_dict[move])
                states_visited += 1
                move_num += 1
                
                # Limpar seen_states periodicamente para não consumir muita memória
                if move_num % 1000 == 0:
                    seen_states.clear()  # Resetar para não consumir muita memória
            
            # Obter métricas do solver
            llm_calls = solver.total_queries
            cache_hits = solver.cache_hits
            
        elif method == "greedy":
            solver = Greedy_Puzzle_Solver(
                ACTIONS_DICT=actions_dict,
                SOLVED_STATE=solved_state,
            )
            
            move_num = 0
            seen_states = set()
            state_history = []
            max_recent_states = 20
            
            while move_num < max_moves:
                # Verificar timeout
                elapsed_time = time.time() - start_time
                if elapsed_time > max_time:
                    break
                
                # Verificar se resolveu
                if current_state == solved_state:
                    success = True
                    break
                
                # Detectar loops
                state_tuple = tuple(current_state)
                seen_states.add(state_tuple)
                state_history.append(state_tuple)
                if len(state_history) > max_recent_states:
                    state_history.pop(0)
                
                # Registrar progresso
                if move_num % 5 == 0:
                    current_correct = sum(1 for s, sol in zip(current_state, solved_state) if s == sol)
                    progress_history.append({
                        "move": move_num,
                        "correct": current_correct,
                        "time": elapsed_time
                    })
                    time_history.append(elapsed_time)
                
                move = solver.choose_action(current_state)
                move_sequence.append(move)
                perform_action(current_state, actions_dict[move])
                states_visited += 1
                move_num += 1
                
                # Limpar memória periodicamente
                if move_num % 1000 == 0:
                    seen_states.clear()
        
        else:
            # Para outros métodos, retornar não implementado
            return {
                "method": method,
                "success": False,
                "moves": 0,
                "time": 0,
                "error": f"Método {method} não implementado neste benchmark"
            }
        
        elapsed_time = time.time() - start_time
        
        # Verificar se realmente resolveu
        final_correct = sum(1 for s, sol in zip(current_state, solved_state) if s == sol)
        actually_solved = final_correct == len(solved_state)
        
        # Calcular métricas de progresso
        max_progress = max([p["correct"] for p in progress_history] + [initial_correct]) if progress_history else initial_correct
        avg_progress = statistics.mean([p["correct"] for p in progress_history]) if progress_history else initial_correct
        
        # Calcular velocidade (peças corretas por segundo)
        progress_rate = (final_correct - initial_correct) / elapsed_time if elapsed_time > 0 else 0
        
        result = {
            "method": method,
            "success": success and actually_solved,
            "moves": len(move_sequence),
            "time": elapsed_time,
            "states_visited": states_visited,
            "initial_correct": initial_correct,
            "final_correct": final_correct,
            "total_positions": len(solved_state),
            "max_progress": max_progress,
            "avg_progress": avg_progress,
            "progress_rate": progress_rate,  # peças corretas por segundo
            "moves_per_second": len(move_sequence) / elapsed_time if elapsed_time > 0 else 0,
            "solution": " ".join(move_sequence) if move_sequence else "",
            "progress_history": progress_history,  # Histórico completo
        }
        
        if method == "llm":
            result["llm_calls"] = llm_calls
            result["cache_hits"] = cache_hits if 'cache_hits' in locals() else 0
            result["llm_calls_per_move"] = llm_calls / len(move_sequence) if move_sequence else 0
            result["cache_hit_rate"] = cache_hits / (llm_calls + cache_hits) * 100 if (llm_calls + cache_hits) > 0 else 0
        
        return result
    
    def run_benchmark(
        self,
        puzzle_name: str,
        methods: List[str] = ["llm", "greedy"],
        num_tests: int = 5,
        scramble_length: int = 10,
        max_moves: int = 10000,  # Continuar até resolver
        max_time: float = 300.0,  # 5 minutos por padrão
        **kwargs
    ) -> List[Dict]:
        """
        Executa benchmark completo para um puzzle.
        
        Args:
            puzzle_name: Nome do puzzle
            methods: Lista de métodos para testar
            num_tests: Número de testes por método
            scramble_length: Comprimento do scramble
            max_moves: Número máximo de movimentos
            **kwargs: Parâmetros adicionais
            
        Returns:
            Lista de resultados
        """
        print(f"\n{'='*70}")
        print(f"Benchmark: {puzzle_name}")
        print(f"{'='*70}")
        
        # Carregar puzzle
        try:
            solved_state, actions_dict = self.load_puzzle_simple(puzzle_name)
            print(f"Puzzle carregado: {len(solved_state)} stickers, {len(actions_dict)} movimentos")
        except Exception as e:
            print(f"Erro ao carregar puzzle: {e}")
            return []
        
        all_results = []
        
        for test_num in range(num_tests):
            print(f"\n--- Teste {test_num + 1}/{num_tests} ---")
            
            # Criar estado embaralhado
            scrambled_state, scramble_moves = self.create_scrambled_state(
                solved_state, actions_dict, scramble_length
            )
            
            correct = sum(1 for s, sol in zip(scrambled_state, solved_state) if s == sol)
            correct_pct = (correct / len(solved_state)) * 100
            print(f"Estado embaralhado: {correct}/{len(solved_state)} corretos ({correct_pct:.1f}%)")
            
            # Testar cada método
            for method in methods:
                print(f"  Testando {method.upper()}...", end=" ", flush=True)
                
                try:
                    result = self.solve_with_method(
                        start_state=scrambled_state.copy(),
                        actions_dict=actions_dict,
                        solved_state=solved_state,
                        method=method,
                        max_moves=max_moves,
                        max_time=max_time,
                        puzzle_name=puzzle_name,
                        last_correct=correct,
                        **kwargs
                    )
                    
                    # Adicionar métricas de cache para LLM
                    if method == "llm" and hasattr(kwargs.get('_solver', None), 'total_queries'):
                        solver = kwargs.get('_solver')
                        result["llm_calls"] = solver.total_queries if solver else 0
                        result["cache_hits"] = solver.cache_hits if solver else 0
                    
                    result.update({
                        "puzzle": puzzle_name,
                        "test_num": test_num + 1,
                        "scramble_length": len(scramble_moves),
                        "initial_correct": correct,
                        "initial_correct_pct": correct_pct,
                    })
                    
                    # Mostrar progresso em tempo real
                    if result.get("success"):
                        print(f"    [OK] Resolvido em {result['moves']} movimentos, {result['time']:.2f}s")
                    else:
                        print(f"    [TIMEOUT] Apos {result['moves']} movimentos, {result['time']:.2f}s")
                        print(f"      Progresso final: {result.get('final_correct', 0)}/{result.get('total_positions', 0)} corretos")
                    
                    all_results.append(result)
                    
                    # Status já foi mostrado acima
                    pass
                    
                except Exception as e:
                    print(f"ERRO: {e}")
                    all_results.append({
                        "puzzle": puzzle_name,
                        "method": method,
                        "test_num": test_num + 1,
                        "success": False,
                        "error": str(e),
                    })
        
        self.results.extend(all_results)
        return all_results
    
    def generate_report(self, format: str = "all") -> Dict[str, str]:
        """
        Gera relatórios dos resultados.
        
        Args:
            format: Formato do relatório ('json', 'csv', 'markdown', 'all')
            
        Returns:
            Dicionário com caminhos dos arquivos gerados
        """
        if not self.results:
            print("Nenhum resultado para gerar relatório")
            return {}
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        files = {}
        
        if format in ["json", "all"]:
            json_path = self.output_dir / f"benchmark_{timestamp}.json"
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(self.results, f, indent=2, ensure_ascii=False)
            files["json"] = str(json_path)
        
        if format in ["csv", "all"]:
            csv_path = self.output_dir / f"benchmark_{timestamp}.csv"
            if self.results:
                fieldnames = self.results[0].keys()
                with open(csv_path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(self.results)
            files["csv"] = str(csv_path)
        
        if format in ["markdown", "all"]:
            md_path = self.output_dir / f"benchmark_{timestamp}.md"
            self._generate_markdown_report(md_path)
            files["markdown"] = str(md_path)
        
        return files
    
    def _generate_markdown_report(self, output_path: Path):
        """Gera relatório em Markdown."""
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("# Benchmark Results - LLM Puzzle Solver\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Agrupar por puzzle e método
            puzzles = {}
            for result in self.results:
                puzzle = result.get("puzzle", "unknown")
                method = result.get("method", "unknown")
                
                if puzzle not in puzzles:
                    puzzles[puzzle] = {}
                if method not in puzzles[puzzle]:
                    puzzles[puzzle][method] = []
                
                puzzles[puzzle][method].append(result)
            
            # Estatísticas por puzzle e método
            for puzzle_name, methods in puzzles.items():
                f.write(f"## {puzzle_name}\n\n")
                
                for method_name, results in methods.items():
                    f.write(f"### {method_name.upper()}\n\n")
                    
                    successful = [r for r in results if r.get("success", False)]
                    if successful:
                        moves = [r["moves"] for r in successful]
                        times = [r["time"] for r in successful]
                        
                        f.write(f"- **Taxa de sucesso**: {len(successful)}/{len(results)} ({len(successful)/len(results)*100:.1f}%)\n")
                        f.write(f"- **Movimentos**: média={statistics.mean(moves):.1f}, mediana={statistics.median(moves):.1f}, min={min(moves)}, max={max(moves)}\n")
                        f.write(f"- **Tempo**: média={statistics.mean(times):.2f}s, mediana={statistics.median(times):.2f}s, min={min(times):.2f}s, max={max(times):.2f}s\n")
                        
                        # Métricas adicionais
                        progress_rates = [r.get("progress_rate", 0) for r in successful if r.get("progress_rate")]
                        moves_per_sec = [r.get("moves_per_second", 0) for r in successful if r.get("moves_per_second")]
                        if progress_rates:
                            f.write(f"- **Taxa de progresso**: média={statistics.mean(progress_rates):.3f} peças/s\n")
                        if moves_per_sec:
                            f.write(f"- **Velocidade**: média={statistics.mean(moves_per_sec):.2f} movimentos/s\n")
                        
                        if method_name == "llm":
                            llm_calls = [r.get("llm_calls", 0) for r in successful]
                            cache_hits = [r.get("cache_hits", 0) for r in successful]
                            if llm_calls:
                                f.write(f"- **Chamadas LLM**: média={statistics.mean(llm_calls):.1f}, total={sum(llm_calls)}\n")
                                if cache_hits:
                                    f.write(f"- **Cache hits**: média={statistics.mean(cache_hits):.1f}, total={sum(cache_hits)}\n")
                                    cache_rate = sum(cache_hits) / (sum(llm_calls) + sum(cache_hits)) * 100 if (sum(llm_calls) + sum(cache_hits)) > 0 else 0
                                    f.write(f"- **Taxa de cache**: {cache_rate:.1f}%\n")
                    else:
                        f.write("- **Taxa de sucesso**: 0%\n")
                    
                    f.write("\n")
            
            # Tabela comparativa
            f.write("## Comparação de Métodos\n\n")
            f.write("| Método | Taxa Sucesso | Movimentos (média) | Tempo (média) |\n")
            f.write("|--------|--------------|---------------------|---------------|\n")
            
            method_stats = {}
            for result in self.results:
                method = result.get("method", "unknown")
                if method not in method_stats:
                    method_stats[method] = {"success": [], "moves": [], "times": []}
                
                if result.get("success"):
                    method_stats[method]["success"].append(True)
                    method_stats[method]["moves"].append(result.get("moves", 0))
                    method_stats[method]["times"].append(result.get("time", 0))
                else:
                    method_stats[method]["success"].append(False)
            
            for method, stats in method_stats.items():
                total = len(stats["success"])
                success_rate = sum(stats["success"]) / total * 100 if total > 0 else 0
                avg_moves = statistics.mean(stats["moves"]) if stats["moves"] else 0
                avg_time = statistics.mean(stats["times"]) if stats["times"] else 0
                
                f.write(f"| {method} | {success_rate:.1f}% | {avg_moves:.1f} | {avg_time:.2f}s |\n")
            
            f.write("\n")
    
    def get_statistics(self) -> Dict:
        """Retorna estatísticas agregadas dos resultados."""
        if not self.results:
            return {}
        
        stats = {}
        
        # Agrupar por método
        by_method = {}
        for result in self.results:
            method = result.get("method", "unknown")
            if method not in by_method:
                by_method[method] = []
            by_method[method].append(result)
        
        for method, results in by_method.items():
            successful = [r for r in results if r.get("success", False)]
            
            stats[method] = {
                "total_tests": len(results),
                "successful": len(successful),
                "success_rate": len(successful) / len(results) * 100 if results else 0,
            }
            
            if successful:
                moves = [r["moves"] for r in successful]
                times = [r["time"] for r in successful]
                
                stats[method].update({
                    "avg_moves": statistics.mean(moves),
                    "median_moves": statistics.median(moves),
                    "min_moves": min(moves),
                    "max_moves": max(moves),
                    "avg_time": statistics.mean(times),
                    "median_time": statistics.median(times),
                    "min_time": min(times),
                    "max_time": max(times),
                })
                
                if method == "llm":
                    llm_calls = [r.get("llm_calls", 0) for r in successful]
                    if llm_calls:
                        stats[method]["avg_llm_calls"] = statistics.mean(llm_calls)
                        stats[method]["total_llm_calls"] = sum(llm_calls)
        
        return stats

