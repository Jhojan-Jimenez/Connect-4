import sys
import os

# --- FIX IMPORTS ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(SCRIPT_DIR)
# -------------------

import numpy as np
from connect4.connect_state import ConnectState
from connect4.policy import Policy
from groups.Magnus_Carlsen.policy import Aha as Magnus


# Bot aleatorio sencillo (equivalente al de Group A, pero definido aquí)
class RandomBot(Policy):
    def mount(self) -> None:
        pass

    def act(self, s: np.ndarray) -> int:
        rng = np.random.default_rng()
        available_cols = [c for c in range(7) if s[0, c] == 0]
        return int(rng.choice(available_cols))


def play_game(policy1: Policy, policy2: Policy, verbose: bool = True) -> int:
    """
    Juega una partida completa entre dos políticas.
    policy1 = jugador ROJO (-1)
    policy2 = jugador AMARILLO (1)
    """
    state = ConnectState()  # Tablero vacío
    policy1.mount()
    policy2.mount()

    if verbose:
        print("Inicio del juego: Magnus (-1) vs RandomBot (1)")

    while not state.is_final():
        if state.player == -1:
            action = policy1.act(state.board)
            if verbose:
                print(f"Magnus juega columna {action}")
        else:
            action = policy2.act(state.board)
            if verbose:
                print(f"RandomBot juega columna {action}")

        state = state.transition(action)

    winner = state.get_winner()

    if verbose:
        state.show()
        if winner == -1:
            print("¡Magnus (MCTS-UCB1) GANÓ!")
        elif winner == 1:
            print("Ganó el RandomBot :(")
        else:
            print("Empate.")

    return winner


def main():
    games = 20
    magnus_wins = 0
    random_wins = 0
    draws = 0

    for i in range(games):
        print(f"\n===== Partida {i+1} =====")
        magnus = Magnus(simulations=200)   # Ajusta simulations si va muy lento
        random_bot = RandomBot()

        result = play_game(magnus, random_bot, verbose=True)

        if result == -1:
            magnus_wins += 1
        elif result == 1:
            random_wins += 1
        else:
            draws += 1

    print("\n===== RESULTADOS =====")
    print(f"Magnus ganó: {magnus_wins}/{games}")
    print(f"RandomBot ganó: {random_wins}/{games}")
    print(f"Empates: {draws}/{games}")


if __name__ == "__main__":
    main()
