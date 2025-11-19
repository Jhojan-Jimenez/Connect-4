import math
import numpy as np
from connect4.policy import Policy
from connect4.connect_state import ConnectState


class Node:
    """
    Nodo sencillo de MCTS para Connect-4 usando UCB1.
    Todas las estadísticas se almacenan desde la perspectiva
    del jugador raíz (el jugador que llama a act()).
    """

    def __init__(
        self,
        state: ConnectState,
        parent: "Node | None" = None,
        parent_action: int | None = None,
    ):
        self.state = state
        self.parent = parent
        self.parent_action = parent_action

        # hijos: acción -> Node
        self.children: dict[int, "Node"] = {}

        # estadísticas
        self.visits: int = 0
        self.value: float = 0.0  # recompensa acumulada para el jugador raíz

        # acciones aún no exploradas desde este nodo
        self.untried_actions: list[int] = state.get_free_cols()

    def is_fully_expanded(self) -> bool:
        return len(self.untried_actions) == 0

    def best_child(self, c_param: float) -> "Node":
        """
        Selecciona al hijo con máximo valor UCB1.
        UCB1 = Q_i / N_i + c * sqrt(ln(N) / N_i)
        """
        assert self.children, "best_child llamado en un nodo sin hijos"

        best_score = -float("inf")
        best_nodes: list["Node"] = []

        for child in self.children.values():
            if child.visits == 0:
                # Fuerza exploración de nodos nunca visitados
                score = float("inf")
            else:
                exploitation = child.value / child.visits
                exploration = c_param * math.sqrt(
                    math.log(self.visits) / child.visits
                )
                score = exploitation + exploration

            if score > best_score:
                best_score = score
                best_nodes = [child]
            elif score == best_score:
                best_nodes.append(child)

        # Desempate aleatorio entre los mejores
        return np.random.default_rng().choice(best_nodes)


class Aha(Policy):
    """
    Agente Connect-4 basado en MCTS + UCB1.
    En cada llamada a act() construye un árbol de búsqueda desde el estado actual,
    simula varias partidas aleatorias (rollouts) y escoge la acción con más visitas.
    """

    def __init__(self, simulations: int = 400, exploration_c: float = math.sqrt(2.0)):
        # Número total de simulaciones desde el estado actual
        self.simulations = simulations
        self.exploration_c = exploration_c
        self.rng = np.random.default_rng()
        self._root_player: int = -1

    def mount(self) -> None:
        # No hay entrenamiento previo, solo dejar listo el RNG si se desea
        pass

    # -------------------------------------------------------------------------
    # Métodos auxiliares para MCTS
    # -------------------------------------------------------------------------

    def _rollout(self, state: ConnectState, root_player: int) -> float:
        """
        Juega una partida aleatoria hasta llegar a un estado terminal.
        Devuelve la recompensa desde la perspectiva del jugador raíz:
          - Victoria  -> 1.0
          - Empate    -> 0.5
          - Derrota   -> 0.0
        """
        current_state = ConnectState(state.board.copy(), state.player)

        while not current_state.is_final():
            free_cols = current_state.get_free_cols()
            if not free_cols:
                break
            col = int(self.rng.choice(free_cols))
            current_state = current_state.transition(col)

        winner = current_state.get_winner()
        if winner == root_player:
            return 1.0
        if winner == 0:
            return 0.5
        return 0.0

    def _mcts(self, root_state: ConnectState, root_player: int) -> int:
        """
        Ejecuta MCTS con selección UCB1 desde root_state y
        devuelve la mejor acción encontrada.
        """
        root = Node(root_state)

        for _ in range(self.simulations):
            node = root

            # 1) SELECCIÓN: bajar por el árbol mientras el nodo esté
            # totalmente expandido y no sea terminal
            while not node.state.is_final() and node.is_fully_expanded():
                node = node.best_child(self.exploration_c)

            # 2) EXPANSIÓN: si no es terminal y aún hay acciones sin probar
            if not node.state.is_final() and node.untried_actions:
                action = int(self.rng.choice(node.untried_actions))
                node.untried_actions.remove(action)
                new_state = node.state.transition(action)
                child = Node(new_state, parent=node, parent_action=action)
                node.children[action] = child
                node = child

            # 3) SIMULACIÓN (ROLLOUT) desde el nodo seleccionado/expandido
            reward = self._rollout(node.state, root_player)

            # 4) BACKPROPAGATION: propagar recompensa hasta la raíz
            while node is not None:
                node.visits += 1
                node.value += reward
                node = node.parent

        # Al final, escoger la acción cuyo hijo tenga más visitas
        if not root.children:
            # Caso raro: sin hijos (no hay movimientos válidos)
            free_cols = root_state.get_free_cols()
            return int(free_cols[0])

        best_child = max(root.children.values(), key=lambda n: n.visits)
        assert best_child.parent_action is not None
        return int(best_child.parent_action)

    # -------------------------------------------------------------------------
    # Interfaz principal de la política
    # -------------------------------------------------------------------------

    def act(self, s: np.ndarray) -> int:
        """
        Recibe el tablero s (6x7) y devuelve la columna (0-6)
        usando MCTS + UCB1 para seleccionar la acción.
        """
        # Deducir de quién es el turno
        num_red = int(np.sum(s == -1))
        num_yellow = int(np.sum(s == 1))
        current_player = -1 if num_red == num_yellow else 1
        self._root_player = current_player

        # Construir el estado del entorno
        current_state = ConnectState(s.copy(), current_player)

        # Casos triviales: sin movimientos o solo uno posible
        free_cols = current_state.get_free_cols()
        if not free_cols:
            # No debería ocurrir si el juego ya se controla con is_final(),
            # pero devolvemos algo válido por seguridad.
            return 0
        if len(free_cols) == 1:
            return int(free_cols[0])

        # Ejecutar MCTS con UCB1 desde el estado actual
        action = self._mcts(current_state, current_player)
        return int(action)
