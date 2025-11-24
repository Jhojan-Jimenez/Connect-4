import math
import numpy as np
import os
import pickle
from connect4.policy import Policy
from connect4.connect_state import ConnectState


#                NODO DEL ÁRBOL PARA MCTS

class Node:
    """
    Nodo que se usa dentro del árbol de búsqueda MCTS.

    En este nodo guardamos:
      - El estado actual del juego (tablero + jugador al que le toca).
      - Un puntero al nodo padre y la acción que se tomó para llegar aquí.
      - Los hijos ya explorados (un hijo por cada acción posible).
      - Estadísticas de visitas y valor acumulado (para UCB1).
      - Las acciones legales que todavía no se han explorado.
    """

    def __init__(self, state, parent=None, parent_action=None):
        # Estado del juego en este nodo
        self.state = state

        # Referencias para poder subir/bajar en el árbol
        self.parent = parent               # nodo anterior en el árbol
        self.parent_action = parent_action # acción que nos trajo desde el padre

        # Hijos: diccionario {acción -> Node}
        self.children = {}

        # Contadores para MCTS
        self.visits = 0                    # cuántas veces se ha visitado este nodo
        self.value = 0.0                   # suma de recompensas que han pasado por aquí

        # Acciones que todavía no se han expandido desde este estado
        # get_free_cols() ya devuelve solo columnas legales
        self.untried_actions = state.get_free_cols()

    def is_fully_expanded(self):
        """
        Devuelve True si ya exploramos todas las acciones legales de este estado.
        """
        return len(self.untried_actions) == 0

    def best_child(self, c_param, Q_table, beta):
        """
        Elige el mejor hijo usando una mezcla entre:

          - PRIOR: valor aprendido Q(s,a) de nuestro diccionario Q_table
          - UCB1: fórmula clásica de MCTS que balancea exploración y explotación

        score = (1 - beta) * PRIOR + beta * UCB1

        donde:
          PRIOR  = Q(s,a) si existe, en otro caso 0.5 (neutro)
          UCB1   = (valor_promedio) + c * sqrt( ln(N_padre) / N_hijo )
        """

        best_score = -float("inf")
        best_nodes = []

        # Convertimos el tablero a tupla para poder usarlo como clave en Q_table
        s_key = tuple(int(x) for x in self.state.board.flatten())

        # Recorremos todos los hijos ya creados
        for action, child in self.children.items():

            # 1) PRIOR: valor aprendido de Q(s,a)
            prior = Q_table.get((s_key, action), 0.5)

            # 2) UCB1 clásico
            if child.visits > 0:
                # Explotación: qué tan bien le ha ido a este hijo
                exploitation = child.value / child.visits
                # Exploración: incentiva visitar nodos menos explorados
                exploration = c_param * math.sqrt(
                    math.log(self.visits) / child.visits
                )
            else:
                # Si nunca se ha visitado, lo forzamos a ser atractivo
                exploitation = float("inf")
                exploration = float("inf")

            ucb1 = exploitation + exploration

            # 3) Mezclamos el prior con UCB1
            score = (1 - beta) * prior + beta * ucb1

            # Nos quedamos con el/los mejores
            if score > best_score:
                best_score = score
                best_nodes = [child]
            elif score == best_score:
                best_nodes.append(child)

        # Si hay empate entre varios hijos, escogemos uno al azar
        return np.random.default_rng().choice(best_nodes)



#                 AGENTE Aha (VERSIÓN HÍBRIDA)

class Aha(Policy):
    """
    Agente para Conecta-4 que mezcla varias ideas:

    - Heurísticas "clásicas" del agente viejo:
        * Si puedo ganar en este turno, juego esa columna.
        * Si el rival puede ganar en su próximo turno, se la bloqueo.
    - Búsqueda MCTS con UCB1:
        * Simulamos muchas partidas aleatorias para estimar qué tan buenas son las jugadas.
    - Q-learning con memoria persistente:
        * Guardamos Q(s,a) en un archivo .pkl para que el agente vaya mejorando
          con el tiempo a medida que juega más partidas.
    """

    def __init__(
        self,
        simulations=150,                 # número de simulaciones por jugada en MCTS
        exploration_c=math.sqrt(2),      # parámetro "c" de UCB1
        alpha=0.3,                       # tasa de aprendizaje para Q-learning
        beta=0.7,                        # mezcla PRIOR vs UCB1
        q_file="magnus_q.pkl",           # archivo donde guardamos la tabla Q
    ):
        self.simulations = simulations
        self.exploration_c = exploration_c
        self.alpha = alpha
        self.beta = beta

        # Random generator propio del agente
        self.rng = np.random.default_rng()

        # Tabla Q y archivo donde se guarda
        self.q_file = q_file
        self.Q = {}   # diccionario donde la clave es (estado, acción)


    #  MOUNT: requerido por la interfaz de la tarea / Gradescope

    def mount(self, *args, **kwargs):
        """
        En Gradescope llaman a policy.mount(timeout).

        Para evitar errores de firma, aceptamos cualquier parámetro,
        pero internamente solo usamos esto para cargar la Q-table
        desde disco (si existe).
        """
        self.load_Q()


    #        CARGA Y GUARDADO DE LA TABLA Q(s,a) EN DISCO

    def load_Q(self):
        """
        Carga la tabla Q desde el archivo q_file, si existe.
        Si no existe o hay algún problema, se inicializa como un diccionario vacío.
        """
        if os.path.exists(self.q_file):
            try:
                with open(self.q_file, "rb") as f:
                    self.Q = pickle.load(f)
            except Exception:
                # Si el archivo está corrupto o falla la carga, reiniciamos Q
                self.Q = {}
        else:
            self.Q = {}

    def save_Q(self):
        """
        Guarda la tabla Q en el archivo q_file.
        En algunos entornos (como el autograder) puede no estar permitido escribir
        en disco, por eso capturamos cualquier excepción y la ignoramos.
        """
        try:
            with open(self.q_file, "wb") as f:
                pickle.dump(self.Q, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception:
            # Si no se puede guardar, simplemente seguimos (el agente igual funciona,
            # solo que no persiste lo aprendido entre ejecuciones).
            pass


    #              HEURÍSTICAS DEL AGENTE "VIEJO"

    def _winning_move(self, state: ConnectState, player: int):
        """
        Recorre todas las columnas legales y revisa si alguna produce
        una victoria inmediata para 'player'.

        Si encuentra una, devuelve esa columna. Si no, devuelve None.
        """
        free = state.get_free_cols()
        for col in free:
            if state.is_applicable(col):
                next_state = state.transition(col)
                if next_state.get_winner() == player:
                    return col
        return None

    def _block_enemy(self, state: ConnectState, player: int):
        """
        Revisa las jugadas del rival: si el oponente puede ganar en su
        próximo turno jugando en alguna columna, la devolvemos para bloquear.
        Si no hay amenaza directa, devolvemos None.
        """
        enemy = -player
        free = state.get_free_cols()
        for col in free:
            # Simulamos que el enemigo juega desde este mismo tablero
            enemy_state = ConnectState(state.board.copy(), enemy)
            if enemy_state.is_applicable(col):
                next_s = enemy_state.transition(col)
                if next_s.get_winner() == enemy:
                    return col
        return None


    #                ROLLOUT ALEATORIO (SIMULACIÓN)

    def _rollout(self, state: ConnectState, root_player: int) -> float:
        """
        A partir de un estado dado, jugamos una partida completamente
        aleatoria hasta que termine.

        Devolvemos:
          - 1.0 si gana el jugador raíz (root_player)
          - 0.5 si hay empate
          - 0.0 si pierde el jugador raíz
        """
        current = ConnectState(state.board.copy(), state.player)

        while not current.is_final():
            free = current.get_free_cols()
            if not free:
                break
            col = int(self.rng.choice(free))
            current = current.transition(col)

        winner = current.get_winner()
        if winner == root_player:
            return 1.0
        if winner == 0:
            return 0.5
        return 0.0


    #        MCTS + PRIORS (Q) + ACTUALIZACIÓN Q-LEARNING

    def _mcts(self, root_state: ConnectState, root_player: int) -> int:
        """
        Ejecuta el ciclo completo de MCTS desde root_state:

          1) Selección: bajamos por el árbol usando best_child() hasta
             llegar a un nodo no expandido o terminal.
          2) Expansión: si el nodo no es terminal y tiene acciones sin usar,
             expandimos una de ellas.
          3) Simulación (rollout): jugamos aleatorio hasta el final.
          4) Backpropagation: propagamos la recompensa hacia arriba,
             actualizando visits y value.

        Además, guardamos (estado, acción, recompensa) en una lista de
        experiencias para luego actualizar la tabla Q(s,a).
        """
        root = Node(root_state)
        experience = []

        # Repetimos el proceso tantas veces como simulaciones hayamos configurado
        for _ in range(self.simulations):
            node = root


            # 1) SELECCIÓN

            while not node.state.is_final() and node.is_fully_expanded():
                node = node.best_child(
                    self.exploration_c,
                    self.Q,
                    self.beta,
                )

            # 2) EXPANSIÓN

            if not node.state.is_final() and node.untried_actions:
                action = int(self.rng.choice(node.untried_actions))
                node.untried_actions.remove(action)

                new_state = node.state.transition(action)
                child = Node(new_state, parent=node, parent_action=action)
                node.children[action] = child
                node = child


            # 3) SIMULACIÓN (ROLLOUT)

            reward = self._rollout(node.state, root_player)


            # 4) BACKPROPAGATION
            #    (y recolección de experiencias para Q-learning)

            while node is not None:
                node.visits += 1
                node.value += reward

                if node.parent is not None:
                    # Guardamos la experiencia desde el padre:
                    # estado del padre, acción que llevó a este nodo y recompensa final
                    s_key = tuple(int(x) for x in node.parent.state.board.flatten())
                    experience.append((s_key, node.parent_action, reward))

                node = node.parent

        # Actualizamos la tabla Q(s,a) con todas las experiencias

        for (s_key, a, r) in experience:
            old_q = self.Q.get((s_key, a), 0.5)
            # Regla clásica de actualización incremental
            self.Q[(s_key, a)] = old_q + self.alpha * (r - old_q)

        # Intentamos guardar en disco lo aprendido
        self.save_Q()


        # Elegimos la acción final: el hijo con más visitas

        if not root.children:
            # Caso raro: si por alguna razón no hay hijos,
            # devolvemos alguna columna libre válida.
            free_cols = root_state.get_free_cols()
            return int(free_cols[0])

        best_child = max(root.children.values(), key=lambda n: n.visits)
        return int(best_child.parent_action)

    #                        ACT (POLICY)

    def act(self, s: np.ndarray) -> int:
        """
        Recibe el tablero s (6x7) y devuelve una columna entre 0 y 6.

        Orden de decisión del agente:
          1) Si hay una jugada que gana ya mismo, la toma.
          2) Si el rival puede ganar en la siguiente, se la bloquea.
          3) Si el estado ya fue visto antes, intenta usar la acción con mejor Q(s,a).
          4) Si nada de lo anterior aplica, corre MCTS y aprende de la simulación.

        En todos los casos se asegura de devolver una jugada legal.
        """

        # Contamos fichas rojas y amarillas para deducir de quién es el turno
        num_red = int(np.sum(s == -1))
        num_yellow = int(np.sum(s == 1))
        current_player = -1 if num_red == num_yellow else 1

        # Creamos el estado a partir del tablero y el jugador actual
        state = ConnectState(s.copy(), current_player)
        free = state.get_free_cols()  # columnas legales

        # Si no hay movimientos posibles (tablero lleno), devolvemos algo válido
        if not free:
            return 0

        # Si solo hay una jugada posible, la tomamos sin pensar más
        if len(free) == 1:
            return int(free[0])

        # 1) Intentamos ganar inmediatamente
        win = self._winning_move(state, current_player)
        if win is not None and win in free and state.is_applicable(win):
            return int(win)

        # 2) Intentamos bloquear una victoria inmediata del rival
        block = self._block_enemy(state, current_player)
        if block is not None and block in free and state.is_applicable(block):
            return int(block)

        # 3) Buscamos si este estado ya existe en la tabla Q
        s_key = tuple(int(x) for x in s.flatten())
        candidates = [(k[1], v) for k, v in self.Q.items() if k[0] == s_key]

        if candidates:
            # Ordenamos las acciones por su valor Q de mayor a menor
            candidates.sort(key=lambda x: x[1], reverse=True)
            best_q_action = candidates[0][0]

            # Si la mejor acción aprendida es legal, la usamos
            if best_q_action in free and state.is_applicable(best_q_action):
                return int(best_q_action)

        # 4) Si nada de lo anterior funcionó, usamos MCTS para decidir
        return self._mcts(state, current_player)
