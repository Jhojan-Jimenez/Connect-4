import sys
import os

# --- FIX IMPORTS (para que funcione desde cualquier ruta) ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(SCRIPT_DIR)
# ------------------------------------------------------------

import numpy as np
from connect4.connect_state import ConnectState
from connect4.policy import Policy

# Importa los Magnus
from groups.Magnus_Old.policy import Aha as MagnusOLD   # AJUSTA si tu carpeta se llama distinto
from groups.Magnus_Carlsen.policy import Aha as MagnusNEW


# Bot auxiliar para imprimir partidas sin fallos
class SilentPolicy(Policy):
    def mount(self): pass
    def act(self, s): raise NotImplementedError()


def play_game(policy_red: Policy, policy_yellow: Policy, verbose=False):
    """
    Simula un juego completo entre dos políticas.
    Rojo  = -1
    Amarillo = +1
    """

    state = ConnectState()
    policy_red.mount()
    policy_yellow.mount()

    while not state.is_final():
        if state.player == -1:
            action = policy_red.act(state.board)
            if verbose:
                print(f"RED juega {action}")
        else:
            action = policy_yellow.act(state.board)
            if verbose:
                print(f"YELLOW juega {action}")

        state = state.transition(action)

    if verbose:
        state.show()

    return state.get_winner()


def main():
    GAMES = 40

    results = {
        "OLD_as_RED": {"old": 0, "new": 0, "draw": 0},
        "NEW_as_RED": {"old": 0, "new": 0, "draw": 0},
    }

    print("\n===================================")
    print("   🔥  TEST 1: OLD (ROJO) vs NEW (AMARILLO)")
    print("===================================\n")

    for i in range(GAMES):
        print(f"Partida {i+1}/{GAMES}")
        winner = play_game(MagnusOLD(simulations=200), MagnusNEW(simulations=200), verbose=False)

        if winner == -1:
            results["OLD_as_RED"]["old"] += 1
        elif winner == 1:
            results["OLD_as_RED"]["new"] += 1
        else:
            results["OLD_as_RED"]["draw"] += 1

    print("\n===================================")
    print("   🔥  TEST 2: NEW (ROJO) vs OLD (AMARILLO)")
    print("===================================\n")

    for i in range(GAMES):
        print(f"Partida {i+1}/{GAMES}")
        winner = play_game(MagnusNEW(simulations=200), MagnusOLD(simulations=200), verbose=False)

        if winner == -1:
            results["NEW_as_RED"]["new"] += 1
        elif winner == 1:
            results["NEW_as_RED"]["old"] += 1
        else:
            results["NEW_as_RED"]["draw"] += 1

    # ===============================
    # 🔥 RESULTADOS FINALES
    # ===============================
    print("\n\n============== RESULTADOS ==============\n")

    print("🔴 OLD (rojo) vs NEW (amarillo):")
    print(f"- Ganó OLD: {results['OLD_as_RED']['old']}")
    print(f"- Ganó NEW: {results['OLD_as_RED']['new']}")
    print(f"- Empates : {results['OLD_as_RED']['draw']}\n")

    print("🔴 NEW (rojo) vs OLD (amarillo):")
    print(f"- Ganó NEW: {results['NEW_as_RED']['new']}")
    print(f>"- Ganó OLD: {results['NEW_as_RED']['old']}")
    print(f"- Empates : {results['NEW_as_RED']['draw']}\n")


    print("==========================================")
    print("   🏆   RESULTADO GLOBAL OLD vs NEW")
    print("==========================================")

    total_old = (
        results["OLD_as_RED"]["old"] +
        results["NEW_as_RED"]["old"]
    )
    total_new = (
        results["OLD_as_RED"]["new"] +
        results["NEW_as_RED"]["new"]
    )
    total_draw = (
        results["OLD_as_RED"]["draw"] +
        results["NEW_as_RED"]["draw"]
    )

    print(f"OLD total wins: {total_old}")
    print(f"NEW total wins: {total_new}")
    print(f"Draws total   : {total_draw}")

    if total_new > total_old:
        print("\n🏆 EL GANADOR FINAL ES: **MAGNUS NEW**")
    elif total_old > total_new:
        print("\n🏆 EL GANADOR FINAL ES: **MAGNUS OLD**")
    else:
        print("\n🤝 EMPATE TOTAL ENTRE OLD Y NEW")


if __name__ == "__main__":
    main()
