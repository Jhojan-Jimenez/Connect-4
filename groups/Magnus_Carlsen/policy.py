import numpy as np
from connect4.policy import Policy
from connect4.connect_state import ConnectState


class Aha(Policy):

    # Simulations: Cantidad de simulaciones por movimiento para escoger la mejor acción.
    def __init__(self, simulations: int = 10):
        self.simulations = simulations
        self.rng = np.random.default_rng()

    def mount(self) -> None:
        pass

    # Simula un juego aleatorio desde el estado dado hasta que termine
    def simulate_random_game(self, state: ConnectState) -> int:

        # Crear una copia del estado para no modificar el original
        board_copy = state.board.copy()
        player_copy = state.player
        current_state = ConnectState(board_copy, player_copy)

        # Mientras no sea un estado final, hacer movimientos aleatorios (greedy)
        # Realiza el juego hasta el final
        while not current_state.is_final():
            available_cols = [c for c in range(
                7) if current_state.board[0, c] == 0]
            if not available_cols:
                break

            # Escoger una columna disponible al azar
            col = int(self.rng.choice(available_cols))

            # Verificar que el movimiento sea válido antes de realizarlo
            if current_state.is_applicable(col):
                current_state = current_state.transition(col)
            else:
                break

        return current_state.get_winner()

    def act(self, s: np.ndarray) -> int:

        # El agente no sabe en que turno esta, por lo que debe deducirlo
        # Para ello debe saber que color es (-1 Rojo, 1 Amarillo)
        num_red = np.sum(s == -1)
        num_yellow = np.sum(s == 1)
        current_player = -1 if num_red == num_yellow else 1

        # Obtiene el estado actual del juego desde la perspectiva del jugador actual
        current_state = ConnectState(s.copy(), current_player)

        # Abtener las columnas disponibles es decir que tengan espacio en la fila superior = 0
        available_cols = [c for c in range(7) if s[0, c] == 0]

        # Situaciones Particulares:
        # Si no hay columnas disponibles, retorna 0 (aunque el juego deberia estar finalizado)
        if not available_cols:
            return 0

        # Si solo hay una columna disponible, retorna esa columna
        if len(available_cols) == 1:
            return available_cols[0]

        # Ejecutar MCTS para cada columna disponible
        scores = {}

        for col in available_cols:
            # Mira si el movimiento es aplicable
            if current_state.is_applicable(col):
                next_state = current_state.transition(col)
            else:
                break

            # Verifica si este movimiento gana inmediatamente, en tal caso returna esa columna
            if next_state.get_winner() == current_player:
                return col

            # Parte Fundamental: Simular juegos aleatorios desde el estado resultante

            wins = 0
            draws = 0
            losses = 0

            # Simular juegos aleatorios desde el estado resultante

            for _ in range(self.simulations):
                winner = self.simulate_random_game(next_state)

                if winner == current_player:
                    wins += 1
                elif winner == 0:
                    draws += 1
                else:
                    losses += 1

            # Sacar un score basado en las simulaciones
            score = wins - losses + (draws * 0.5)
            scores[col] = score

        # Escoger la columna con el mejor puntaje
        if not scores:
            # En caso de que no haya scores, retorna la primera columna disponible
            return int(available_cols[0])

        best_col = max(scores, key=scores.get)
        return int(best_col)
