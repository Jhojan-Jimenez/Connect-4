import sys
import os

# --- FIX IMPORTS ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.append(SCRIPT_DIR)
# -------------------

import numpy as np
from connect4.connect_state import ConnectState
from connect4.policy import Policy

# Importar versión vieja y versión nueva
from groups.Magnus_Old.policy import Aha as MagnusOld   # versión anterior
from groups.Magnus_Carlsen.policy import Aha as MagnusNew  # versión actual (MCTS+UCB1+Q)


# Bot aleatorio sencillo (equivalente al de Group A, pero definido aquí)
class RandomBot(Policy):
    def mount(self) -> None:
        pass

    def act(self, s: np.ndarray) -> int:
        rng = np.random.default_rng()
        available_cols = [c for c in range(7) if s[0, c] == 0]
        return int(rng.choice(available_cols))


def play_game(policy1: Policy, policy2: Policy, name1: str, name2: str, verbose: bool = False) -> int:
    """
    Juega una partida completa entre dos políticas.
    policy1 = jugador ROJO (-1)
    policy2 = jugador AMARILLO (1)
    """
    state = ConnectState()  # Tablero vacío
    policy1.mount()
    policy2.mount()

    if verbose:
        print(f"Inicio del juego: {name1} (-1) vs {name2} (1)")

    while not state.is_final():
        if state.player == -1:
            action = policy1.act(state.board)
            if verbose:
                print(f"{name1} juega columna {action}")
        else:
            action = policy2.act(state.board)
            if verbose:
                print(f"{name2} juega columna {action}")

            # Podrías imprimir el tablero aquí si quisieras

        state = state.transition(action)

    winner = state.get_winner()

    if verbose:
        state.show()
        if winner == -1:
            print(f"¡{name1} GANÓ!")
        elif winner == 1:
            print(f"¡{name2} GANÓ!")
        else:
            print("Empate.")

    return winner


def evaluate_against_random(PolicyClass, name: str, games: int = 20, simulations: int = 200):
    wins = 0
    losses = 0
    draws = 0

    for i in range(games):
        print(f"\n===== {name} vs RandomBot — Partida {i+1}/{games} =====")
        # Instancia de la policy (vieja o nueva)
        # Para la nueva, ignorará parámetros que no existan si la firma es distinta
        try:
            agent = PolicyClass(simulations=simulations)
        except TypeError:
            # Por si la versión nueva tiene más parámetros obligatorios,
            # puedes ajustar aquí si hace falta
            agent = PolicyClass()

        random_bot = RandomBot()

        # Siempre: agent = ROJO (-1), random = AMARILLO (1)
        result = play_game(agent, random_bot, name, "RandomBot", verbose=False)

        if result == -1:
            wins += 1
        elif result == 1:
            losses += 1
        else:
            draws += 1

    print(f"\n===== RESULTADOS {name} vs RandomBot =====")
    print(f"Ganadas por {name}: {wins}/{games}")
    print(f"Ganadas por RandomBot: {losses}/{games}")
    print(f"Empates: {draws}/{games}")
    win_rate = wins / games if games > 0 else 0.0
    print(f"Win rate de {name}: {win_rate:.3f}")
    return wins, losses, draws, win_rate


def main():
    games = 200

    print("\n\n############################")
    print("### 1) Magnus OLD vs Random")
    print("############################")
    evaluate_against_random(MagnusOld, "Magnus_OLD", games=games, simulations=100)

    print("\n\n############################")
    print("### 2) Magnus NEW vs Random")
    print("############################")
    # OJO: aquí asumo que MagnusNew acepta parámetro simulations
    evaluate_against_random(MagnusNew, "Magnus_NEW", games=games, simulations=200)


if __name__ == "__main__":
    main()
