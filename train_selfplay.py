import sys
import os

# --- FIX IMPORTS PARA EJECUTAR DESDE CONSOLA ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.append(SCRIPT_DIR)
# ------------------------------------------------

import numpy as np
from connect4.connect_state import ConnectState
from connect4.policy import Policy
from groups.Magnus_Carlsen.policy import Aha as Magnus


# ------------------------------------------------
#   Bot aleatorio sencillo (oponente baseline)
# ------------------------------------------------
class RandomBot(Policy):
    def mount(self) -> None:
        pass

    def act(self, s: np.ndarray) -> int:
        rng = np.random.default_rng()
        available_cols = [c for c in range(7) if s[0, c] == 0]
        return int(rng.choice(available_cols))


# ------------------------------------------------
#   Función para jugar UNA partida entre 2 agentes
# ------------------------------------------------
def play_single_game(agent_first: Policy, agent_second: Policy, verbose: bool = False) -> int:
    """
    Juega una partida completa entre dos políticas.
    agent_first  = jugador ROJO (-1)
    agent_second = jugador AMARILLO (1)

    Devuelve:
        -1 si gana rojo
         1 si gana amarillo
         0 si hay empate
    """
    state = ConnectState()  # tablero vacío

    # mount solo por si algún agente lo requiere (Magnus carga Q en mount)
    agent_first.mount()
    agent_second.mount()

    while not state.is_final():
        if state.player == -1:  # turno del rojo
            action = agent_first.act(state.board)
        else:                   # turno del amarillo
            action = agent_second.act(state.board)

        state = state.transition(action)

    winner = state.get_winner()

    if verbose:
        state.show()
        if winner == -1:
            print("Ganó ROJO")
        elif winner == 1:
            print("Ganó AMARILLO")
        else:
            print("Empate")

    return winner


# ------------------------------------------------
#   Entrenamiento principal
# ------------------------------------------------
def train(episodes: int = 300, mode: str = "vs_random", report_every: int = 50):
    """
    Entrena a Magnus en diferentes modos.

    Parámetros:
        episodes: número total de episodios de entrenamiento
        mode:
            - "vs_random": Magnus vs bot aleatorio
            # Podrías extender a "selfplay" si luego quieres Magnus vs Magnus
        report_every: cada cuántos episodios mostrar métricas
    """

    # Agente que APRENDE
    magnus = Magnus(
        simulations=200,      # puedes subir/bajar según velocidad
        exploration_c=np.sqrt(2),
        alpha=0.3,
        beta=0.7,
        q_file="magnus_q.pkl",
    )
    # Cargar Q existente (si hay)
    magnus.mount()

    rng = np.random.default_rng()

    # Estadísticas globales
    total_wins_magnus = 0
    total_losses_magnus = 0
    total_draws = 0

    wins_as_first = 0
    wins_as_second = 0

    if mode == "vs_random":
        opponent = RandomBot()
    else:
        raise ValueError(f"Modo de entrenamiento no soportado: {mode}")

    for episode in range(1, episodes + 1):

        # Decidir aleatoriamente quién va primero
        magnus_is_first = rng.random() < 0.5

        if mode == "vs_random":
            if magnus_is_first:
                first_agent = magnus
                second_agent = opponent
            else:
                first_agent = opponent
                second_agent = magnus
        else:
            # Si en el futuro implementas self-play Magnus vs Magnus,
            # aquí podrías crear otra instancia con parámetros diferentes.
            first_agent = magnus
            second_agent = magnus

        # Jugar una partida
        winner = play_single_game(first_agent, second_agent, verbose=False)

        # Interpretar resultado desde perspectiva de Magnus
        if winner == 0:
            total_draws += 1
        else:
            # ROJO = first_agent, AMARILLO = second_agent
            if magnus_is_first:
                # Magnus es ROJO (-1)
                if winner == -1:
                    total_wins_magnus += 1
                    wins_as_first += 1
                elif winner == 1:
                    total_losses_magnus += 1
            else:
                # Magnus es AMARILLO (1)
                if winner == 1:
                    total_wins_magnus += 1
                    wins_as_second += 1
                elif winner == -1:
                    total_losses_magnus += 1

        # Reporte periódico
        if episode % report_every == 0:
            total_played = total_wins_magnus + total_losses_magnus + total_draws
            win_rate = total_wins_magnus / total_played if total_played > 0 else 0.0

            print(f"\n===== Episodio {episode}/{episodes} =====")
            print(f"Ganadas Magnus: {total_wins_magnus}")
            print(f"  - Como primero (rojo):   {wins_as_first}")
            print(f"  - Como segundo (amarillo): {wins_as_second}")
            print(f"Empates: {total_draws}")
            print(f"Perdidas contra random: {total_losses_magnus}")
            print(f"Win rate aproximado: {win_rate:.3f}")
            print(f"Estados aprendidos (|Q|): {len(magnus.Q)}")
            print("Q-table almacenada en:", magnus.q_file)

    print("\n===== ENTRENAMIENTO FINALIZADO =====")
    total_played = total_wins_magnus + total_losses_magnus + total_draws
    win_rate = total_wins_magnus / total_played if total_played > 0 else 0.0

    print(f"Episodios totales: {total_played}")
    print(f"Ganadas: {total_wins_magnus}")
    print(f"Perdidas: {total_losses_magnus}")
    print(f"Empates: {total_draws}")
    print(f"Win rate final: {win_rate:.3f}")
    print(f"Estados almacenados en Q: {len(magnus.Q)}")
    print("Archivo de conocimiento:", magnus.q_file)


if __name__ == "__main__":
    # Puedes ajustar los parámetros aquí
    train(
        episodes=300,
        mode="vs_random",
        report_every=50,
    )
