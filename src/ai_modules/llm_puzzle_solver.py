"""
LLM-based puzzle solver using Ollama for move selection.
This solver uses a language model to suggest the next move based on the current puzzle state.
"""
import random
import json
from typing import Optional
import requests

if __name__ != "__main__":
    from .twisty_puzzle_model import perform_action
else:
    from twisty_puzzle_model import perform_action


class LLM_Puzzle_Solver:
    """
    A puzzle solver that uses an LLM (via Ollama) to suggest moves.
    The LLM receives a textual representation of the puzzle state and suggests the next move.
    """
    
    def __init__(
        self,
        ACTIONS_DICT: dict,
        SOLVED_STATE: list[int],
        ollama_model: str = "llama3.2",
        ollama_url: str = "http://localhost:11434/api/generate",
        puzzle_name: Optional[str] = None,
        cache_responses: bool = True,
    ):
        """
        Initialize the LLM puzzle solver.
        
        Args:
            ACTIONS_DICT: Dictionary containing all possible actions as cycles
            SOLVED_STATE: The solved state as a list of integers
            ollama_model: Name of the Ollama model to use
            ollama_url: URL of the Ollama API endpoint
            puzzle_name: Optional name of the puzzle
            cache_responses: Whether to cache LLM responses for the same state
        """
        self.ACTIONS_DICT = ACTIONS_DICT
        self.ACTION_KEYS = list(ACTIONS_DICT.keys())
        self.N_ACTIONS = len(self.ACTION_KEYS)
        self.SOLVED_STATE = SOLVED_STATE
        self.ollama_model = ollama_model
        self.ollama_url = ollama_url
        self.puzzle_name = puzzle_name or "twisty_puzzle"
        self.cache_responses = cache_responses
        self.response_cache = {}
        self.total_queries = 0  # Contador de queries ao LLM
        self.cache_hits = 0  # Contador de cache hits
        
        # Filter out rotation moves for simpler puzzles (keep only base moves)
        self.base_actions = [
            key for key in self.ACTION_KEYS 
            if not key.startswith("rot_") and not key.startswith("alg_")
        ]
        if not self.base_actions:
            self.base_actions = self.ACTION_KEYS
    
    def state_to_text(self, state: list[int]) -> str:
        """
        Convert puzzle state to a textual representation for the LLM.
        Uses a simplified representation showing color distribution.
        
        Args:
            state: Current puzzle state as list of integers
            
        Returns:
            Textual description of the state
        """
        # Count colors
        color_counts = {}
        for color_idx in state:
            color_counts[color_idx] = color_counts.get(color_idx, 0) + 1
        
        # Count correct positions
        correct_positions = sum(1 for i, (s, sol) in enumerate(zip(state, self.SOLVED_STATE)) if s == sol)
        total_positions = len(state)
        correct_percentage = (correct_positions / total_positions) * 100
        
        # Create description
        state_desc = f"Puzzle state: {total_positions} positions total. "
        state_desc += f"{correct_positions} positions ({correct_percentage:.1f}%) are correct. "
        state_desc += f"Color distribution: {dict(color_counts)}. "
        
        return state_desc
    
    def get_available_moves_text(self) -> str:
        """Get a textual description of available moves."""
        moves_text = "Available moves: " + ", ".join(self.base_actions[:20])  # Limit to first 20
        if len(self.base_actions) > 20:
            moves_text += f" (and {len(self.base_actions) - 20} more)"
        return moves_text
    
    def query_ollama(self, prompt: str) -> str:
        """
        Query Ollama API for a response.
        
        Args:
            prompt: The prompt to send to the LLM
            
        Returns:
            The LLM's response text
        """
        # Check cache
        if self.cache_responses and prompt in self.response_cache:
            self.cache_hits += 1
            return self.response_cache[prompt]
        
        # Increment query counter
        self.total_queries += 1
        
        try:
            response = requests.post(
                self.ollama_url,
                json={
                    "model": self.ollama_model,
                    "prompt": prompt,
                    "stream": False,
                },
                timeout=30,
            )
            response.raise_for_status()
            result = response.json()
            llm_response = result.get("response", "").strip()
            
            # Cache the response
            if self.cache_responses:
                self.response_cache[prompt] = llm_response
                
            return llm_response
        except requests.exceptions.RequestException as e:
            print(f"Error querying Ollama: {e}")
            print("Falling back to random move selection.")
            return ""
    
    def parse_llm_move(self, llm_response: str) -> Optional[str]:
        """
        Parse the LLM response to extract a move name.
        
        Args:
            llm_response: The LLM's response text
            
        Returns:
            Move name if found, None otherwise
        """
        llm_response = llm_response.lower().strip()
        
        # Try to find a move name in the response
        for move in self.base_actions:
            if move.lower() in llm_response:
                return move
        
        # Try to find partial matches (e.g., "R" matches "R", "R'", "R2")
        for move in self.base_actions:
            move_base = move.replace("'", "").replace("2", "").replace("3", "").lower()
            if move_base in llm_response and len(move_base) > 0:
                return move
        
        return None
    
    def _get_puzzle_type_hint(self) -> str:
        """Detect puzzle type and return appropriate hint."""
        puzzle_name = self.puzzle_name.lower()
        
        if "4d" in puzzle_name or "hyper" in puzzle_name:
            if "3x3x3x3" in puzzle_name:
                return """This is a 4D 3x3x3x3 hypercube - a very complex puzzle. Strategy:
- Work systematically, focusing on one dimension at a time
- Prioritize moves that fix multiple pieces simultaneously
- Avoid moves that create new problems
- Think about which pieces are furthest from their solved positions
- Make moves that maximize the number of correct pieces"""
            else:
                return """This is a 4D puzzle. Focus on:
- Understanding which pieces are out of place
- Making moves that bring pieces closer to their solved positions
- Avoiding moves that undo previous progress
- Working dimension by dimension when possible"""
        elif "3x3" in puzzle_name or "cube_3" in puzzle_name:
            return """This is a 3x3 Rubik's cube. Common strategies:
- Solve layer by layer (bottom to top)
- Orient corners before positioning them
- Use algorithms to swap pieces without affecting others
- Focus on completing one face/layer at a time"""
        elif "2x2" in puzzle_name or "cube_2" in puzzle_name:
            return """This is a 2x2 cube. Strategy:
- Focus on getting all corners in correct positions
- Orient corners correctly
- Fewer pieces means each move has more impact
- Complete one face first, then the opposite face"""
        else:
            return """General twisty puzzle solving:
- Maximize the number of correct pieces
- Avoid moves that break already-correct pieces
- Work systematically toward the solved state
- Prioritize moves that fix the most pieces"""
    
    def choose_action(self, state: list[int], max_attempts: int = 3, use_hybrid: bool = True) -> str:
        """
        Choose the next action using the LLM.
        
        Args:
            state: Current puzzle state
            max_attempts: Maximum number of attempts to get a valid move from LLM
            
        Returns:
            The name of the chosen action
        """
        # Build prompt for LLM
        state_text = self.state_to_text(state)
        moves_text = self.get_available_moves_text()
        puzzle_hint = self._get_puzzle_type_hint()
        
        # Count progress
        correct = sum(1 for s, sol in zip(state, self.SOLVED_STATE) if s == sol)
        total = len(self.SOLVED_STATE)
        progress = (correct / total) * 100
        
        # Calcular qual movimento maximiza peças corretas (para ajudar o LLM)
        best_move_hint = ""
        if len(self.base_actions) <= 30:  # Para puzzles pequenos/médios
            move_scores = {}
            sample_size = min(15, len(self.base_actions))  # Testar até 15 movimentos
            import random
            moves_to_test = random.sample(self.base_actions, sample_size) if len(self.base_actions) > sample_size else self.base_actions
            
            for move in moves_to_test:
                test_state = state.copy()
                perform_action(test_state, self.ACTIONS_DICT[move])
                test_correct = sum(1 for s, sol in zip(test_state, self.SOLVED_STATE) if s == sol)
                move_scores[move] = test_correct
            
            if move_scores:
                best_moves = sorted(move_scores.items(), key=lambda x: x[1], reverse=True)[:5]
                best_move_hint = f"\nHint: Top moves that maximize correct pieces: {', '.join([f'{m}(+{c-correct})' for m, c in best_moves[:3]])}"
        
        prompt = f"""You are an expert at solving twisty puzzles. Your goal is to solve this puzzle efficiently.

{puzzle_hint}

Current puzzle state:
{state_text}

{moves_text}
{best_move_hint}

Progress: {correct}/{total} pieces correct ({progress:.1f}%)

Your task: Analyze the current state and choose the SINGLE BEST move that will:
1. Maximize the number of correct pieces after the move, OR
2. Set up pieces for future moves without breaking current progress, OR  
3. Avoid breaking already-correct pieces

Think step by step:
- Which pieces are in wrong positions?
- Which move fixes the most pieces?
- Which move doesn't break correct pieces?

Respond with ONLY the move name (e.g., "R", "U'", "F2"). No explanation, just the move.

Move:"""
        
        # Estratégia híbrida: para puzzles complexos, combinar LLM com análise local
        if use_hybrid and len(self.SOLVED_STATE) > 30:
            # Para puzzles grandes, usar LLM mas validar com análise local
            best_local_moves = self._get_best_local_moves(state, top_n=3)
            
            # Tentar LLM primeiro
            for attempt in range(max_attempts):
                llm_response = self.query_ollama(prompt)
                move = self.parse_llm_move(llm_response)
                
                if move and move in self.ACTION_KEYS:
                    # Validar se o movimento do LLM está entre os melhores locais
                    if move in [m for m, _ in best_local_moves]:
                        return move
                    # Se não estiver, ainda usar mas com menos confiança
                    return move
            
            # Se LLM falhar, usar melhor movimento local
            if best_local_moves:
                return best_local_moves[0][0]
        else:
            # Para puzzles pequenos, usar LLM normalmente
            for attempt in range(max_attempts):
                llm_response = self.query_ollama(prompt)
                move = self.parse_llm_move(llm_response)
                
                if move and move in self.ACTION_KEYS:
                    return move
        
        # Fallback: use greedy approach if LLM fails
        return self._greedy_fallback(state)
    
    def _get_best_local_moves(self, state: list[int], top_n: int = 3) -> list[tuple]:
        """
        Retorna os top N movimentos que maximizam peças corretas.
        
        Returns:
            Lista de tuplas (move_name, correct_count) ordenada por melhor
        """
        move_scores = []
        for move in self.base_actions:
            test_state = state.copy()
            perform_action(test_state, self.ACTIONS_DICT[move])
            correct = sum(1 for s, sol in zip(test_state, self.SOLVED_STATE) if s == sol)
            move_scores.append((move, correct))
        
        # Ordenar por número de peças corretas (maior primeiro)
        move_scores.sort(key=lambda x: x[1], reverse=True)
        return move_scores[:top_n]
    
    def _greedy_fallback(self, state: list[int]) -> str:
        """
        Fallback to greedy selection if LLM fails.
        Chooses move that maximizes correct positions.
        
        Args:
            state: Current puzzle state
            
        Returns:
            Move name
        """
        best_actions = []
        best_correct = -1
        
        for action in self.base_actions:
            new_state = state.copy()
            perform_action(new_state, self.ACTIONS_DICT[action])
            correct = sum(1 for s, sol in zip(new_state, self.SOLVED_STATE) if s == sol)
            
            if correct > best_correct:
                best_actions = [action]
                best_correct = correct
            elif correct == best_correct:
                best_actions.append(action)
        
        return random.choice(best_actions) if best_actions else random.choice(self.base_actions)
    
    def get_state_value(self, state: tuple) -> float:
        """
        Get state value for compatibility with A* solver.
        Uses percentage of correct positions as value.
        
        Args:
            state: Puzzle state as tuple
            
        Returns:
            Value between 0 and 1 (higher is better)
        """
        state_list = list(state)
        correct = sum(1 for s, sol in zip(state_list, self.SOLVED_STATE) if s == sol)
        return correct / len(self.SOLVED_STATE) if self.SOLVED_STATE else 0.0


def solve_with_llm(
    start_state: list[int],
    ACTIONS_DICT: dict,
    SOLVED_STATE: list[int],
    max_moves: int = 100,
    ollama_model: str = "llama3.2",
    verbose: bool = True,
) -> tuple[list[str], bool]:
    """
    Solve a puzzle using LLM-based move selection.
    
    Args:
        start_state: Initial scrambled state
        ACTIONS_DICT: Dictionary of available actions
        SOLVED_STATE: The solved state
        max_moves: Maximum number of moves to attempt
        ollama_model: Ollama model to use
        verbose: Whether to print progress
        
    Returns:
        Tuple of (list of moves, success boolean)
    """
    solver = LLM_Puzzle_Solver(
        ACTIONS_DICT=ACTIONS_DICT,
        SOLVED_STATE=SOLVED_STATE,
        ollama_model=ollama_model,
    )
    
    current_state = start_state.copy()
    move_sequence = []
    
    for move_num in range(max_moves):
        # Check if solved
        if current_state == SOLVED_STATE:
            if verbose:
                print(f"Solved in {len(move_sequence)} moves!")
            return move_sequence, True
        
        # Get next move from LLM
        if verbose:
            print(f"Move {move_num + 1}/{max_moves}: Querying LLM...", end=" ")
        
        move = solver.choose_action(current_state)
        move_sequence.append(move)
        
        # Apply move
        perform_action(current_state, ACTIONS_DICT[move])
        
        if verbose:
            correct = sum(1 for s, sol in zip(current_state, SOLVED_STATE) if s == sol)
            total = len(SOLVED_STATE)
            print(f"Selected: {move} ({correct}/{total} correct)")
    
    if verbose:
        print(f"Failed to solve in {max_moves} moves.")
    return move_sequence, False


if __name__ == "__main__":
    # Test with a simple puzzle
    print("LLM Puzzle Solver - Test Mode")
    print("Make sure Ollama is running on localhost:11434")
    print("Install with: pip install requests")
