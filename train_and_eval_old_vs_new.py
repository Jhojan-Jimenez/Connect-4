import sys
import os

# --- FIX IMPORTS (para que funcione desde cualquier ruta) ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(SCRIPT_DIR)
# ------------------------------------------------------------

import numpy as np
from connect4.connect_state import ConnectState
from connect4.policy import Policy

# ⬅️ AJUSTA ESTOS IMPORTS A TUS RUTAS REALES
from groups.Magnus_Old.policy import Aha as MagnusOLD           # agente viejo
from groups.Magnus_Carlsen.policy import AhaSupreme as MagnusNEW  # agente nuevo (con Q-learning)


def play_game(policy_red: Policy, policy_yellow: Policy, verbose: bool = False) -> int:
    """
    Simula un juego completo entre dos políticas.
    Rojo  = -1
    Amarillo = +1
    Retorna: -1 si gana ROJO, 1 si gana AMARILLO, 0 si hay empate.
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
    # 🔧 HIPERPARÁMETROS DE ENTRENAMIENTO
    TOTAL_GAMES = 400          # súbelo a 1000+ cuando veas que va bien
    SIM_OLD = 150              # simulaciones del agente viejo
    SIM_NEW = 250              # simulaciones del nuevo (más fuerte / más lento)

    # Estadísticas separadas por rol
    results = {
        "OLD_as_RED": {"old": 0, "new": 0, "draw": 0},
        "NEW_as_RED": {"old": 0, "new": 0, "draw": 0},
    }

    print("\n===================================")
    print("   🔥 ENTRENAMIENTO + COMPETENCIA")
    print("   MagnusNEW (MCTS + Q) vs MagnusOLD")
    print("===================================\n")

    for game_idx in range(1, TOTAL_GAMES + 1):
        # 🔁 alternar colores:
        # juegos impares -> OLD rojo, NEW amarillo
        # juegos pares   -> NEW rojo, OLD amarillo
        if game_idx % 2 == 1:
            mode = "OLD_as_RED"
            red_agent = MagnusOLD(simulations=SIM_OLD)
            yellow_agent = MagnusNEW(simulations=SIM_NEW)
        else:
            mode = "NEW_as_RED"
            red_agent = MagnusNEW(simulations=SIM_NEW)
            yellow_agent = MagnusOLD(simulations=SIM_OLD)

        print(f"Partida {game_idx}/{TOTAL_GAMES}  ({mode})")

        winner = play_game(red_agent, yellow_agent, verbose=False)

        if mode == "OLD_as_RED":
            if winner == -1:
                results["OLD_as_RED"]["old"] += 1
            elif winner == 1:
                results["OLD_as_RED"]["new"] += 1
            else:
                results["OLD_as_RED"]["draw"] += 1
        else:  # NEW_as_RED
            if winner == -1:
                results["NEW_as_RED"]["new"] += 1
            elif winner == 1:
                results["NEW_as_RED"]["old"] += 1
            else:
                results["NEW_as_RED"]["draw"] += 1

        # 🧠 IMPORTANTE:
        # Cada vez que MagnusNEW hace un movimiento, internamente:
        #  - llama a _mcts(...)
        #  - recolecta experiencias (s,a,r)
        #  - actualiza Q[(s,a)]
        #  - guarda magnus_q.pkl
        # => O sea: CADA PARTIDA es entrenamiento.

        # Reporte parcial cada 20 partidas
        if game_idx % 20 == 0:
            total_old = (
                results["OLD_as_RED"]["old"] + results["NEW_as_RED"]["old"]
            )
            total_new = (
                results["OLD_as_RED"]["new"] + results["NEW_as_RED"]["new"]
            )
            total_draw = (
                results["OLD_as_RED"]["draw"] + results["NEW_as_RED"]["draw"]
            )
            total_played = total_old + total_new + total_draw
            winrate_new = total_new / total_played if total_played > 0 else 0.0

            print("\n---- RESUMEN PARCIAL ----")
            print(f"Partidas jugadas: {total_played}/{TOTAL_GAMES}")
            print(f"OLD wins totales : {total_old}")
            print(f"NEW wins totales : {total_new}")
            print(f"Empates totales  : {total_draw}")
            print(f"Win rate NEW ≈ {winrate_new:.3f}\n")
            print("-------------------------\n")

    # ===============================
    # 🔥 RESULTADOS FINALES
    # ===============================
    print("\n\n============== RESULTADOS FINALES ==============\n")

    print("🔴 OLD (rojo) vs NEW (amarillo):")
    print(f"- Ganó OLD: {results['OLD_as_RED']['old']}")
    print(f"- Ganó NEW: {results['OLD_as_RED']['new']}")
    print(f"- Empates : {results['OLD_as_RED']['draw']}\n")

    print("🔴 NEW (rojo) vs OLD (amarillo):")
    print(f"- Ganó NEW: {results['NEW_as_RED']['new']}")
    print(f"- Ganó OLD: {results['NEW_as_RED']['old']}")
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
