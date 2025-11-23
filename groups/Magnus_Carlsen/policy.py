import math
import numpy as np
import os
import pickle
from connect4.policy import Policy
from connect4.connect_state import ConnectState


class Node:
    def __init__(self, state: ConnectState, parent=None, parent_action=None):
        self.state = state
        self.parent = parent
        self.parent_action = parent_action

        self.children = {}
        self.visits = 0
        self.value = 0.0

        self.untried_actions = state.get_free_cols()

    def is_fully_expanded(self):
        return len(self.untried_actions) == 0

    def best_child(self, c_param, Q_table=None, beta=0.7):
        best_score = -float("inf")
        best_nodes = []
        s_key = tuple(int(x) for x in self.state.board.flatten())

        for action, child in self.children.items():

            # PRIOR Q(s,a)
            prior = Q_table.get((s_key, action), 0.5) if Q_table else 0.5

            # EXPLOTATION
            if child.visits > 0:
                exploitation = child.value / child.visits
                exploration = c_param * math.sqrt(math.log(self.visits) / child.visits)
            else:
                exploitation = float("inf")
                exploration = float("inf")

            ucb1 = exploitation + exploration

            # Mezcla prior y UCB1
            score = (1 - beta) * prior + beta * ucb1

            if score > best_score:
                best_score = score
                best_nodes = [child]
            elif score == best_score:
                best_nodes.append(child)

        return np.random.default_rng().choice(best_nodes)


class Aha(Policy):

    def __init__(
        self,
        simulations=250,
        exploration_c=math.sqrt(2),
        alpha=0.3,
        beta=0.7,
        q_file="magnus_q.pkl",
    ):
        self.simulations = simulations
        self.exploration_c = exploration_c
        self.alpha = alpha
        self.beta = beta
        self.rng = np.random.default_rng()

        self.q_file = q_file
        self.Q = {}  # Q(s,a)

        self._root_player = -1

    # ==========================
    # MANEJO DE Q EN BINARIO
    # ==========================
    def load_Q(self):
        if os.path.exists(self.q_file):
            with open(self.q_file, "rb") as f:
                self.Q = pickle.load(f)
        else:
            self.Q = {}

    def save_Q(self):
        with open(self.q_file, "wb") as f:
            pickle.dump(self.Q, f, protocol=pickle.HIGHEST_PROTOCOL)

    def mount(self):
        self.load_Q()

    # ==========================
    #      ROLLOUT
    # ==========================
    def _rollout(self, state: ConnectState, root_player: int):
        current = ConnectState(state.board.copy(), state.player)
        while not current.is_final():
            free = current.get_free_cols()
            col = int(self.rng.choice(free))
            current = current.transition(col)

        winner = current.get_winner()
        if winner == root_player:
            return 1.0
        if winner == 0:
            return 0.5
        return 0.0

    # ==========================
    #   MCTS + Q-LEARNING
    # ==========================
    def _mcts(self, root_state, root_player):
        root = Node(root_state)
        experience = []

        for _ in range(self.simulations):
            node = root

            # SELECCIÓN
            while not node.state.is_final() and node.is_fully_expanded():
                node = node.best_child(
                    self.exploration_c,
                    Q_table=self.Q,
                    beta=self.beta,
                )

            # EXPANSIÓN
            if not node.state.is_final() and node.untried_actions:
                action = int(self.rng.choice(node.untried_actions))
                node.untried_actions.remove(action)
                new_state = node.state.transition(action)

                child = Node(new_state, parent=node, parent_action=action)
                node.children[action] = child
                node = child

            # SIMULACIÓN
            reward = self._rollout(node.state, root_player)

            # BACKPROP + almacenar experiencias
            while node is not None:
                node.visits += 1
                node.value += reward

                if node.parent is not None:
                    s_key = tuple(int(x) for x in node.parent.state.board.flatten())
                    experience.append((s_key, node.parent_action, reward))

                node = node.parent

        # Q-learning
        for (s_key, a, r) in experience:
            old_q = self.Q.get((s_key, a), 0.5)
            self.Q[(s_key, a)] = old_q + self.alpha * (r - old_q)

        self.save_Q()

        # escoger mov con más visitas
        best_child = max(root.children.values(), key=lambda n: n.visits)
        return int(best_child.parent_action)

    # ==========================
    #       ACT
    # ==========================
    def act(self, s: np.ndarray) -> int:

        num_red = int(np.sum(s == -1))
        num_yellow = int(np.sum(s == 1))
        current_player = -1 if num_red == num_yellow else 1
        self._root_player = current_player

        state = ConnectState(s.copy(), current_player)
        free = state.get_free_cols()

        if len(free) == 1:
            return int(free[0])

        # fallback si el estado ya existe en Q
        s_key = tuple(int(x) for x in s.flatten())
        candidates = [(k[1], v) for k, v in self.Q.items() if k[0] == s_key]

        if candidates:
            candidates.sort(key=lambda x: x[1], reverse=True)
            best_action = candidates[0][0]
            if best_action in free:
                return int(best_action)

        # si no está → MCTS
        return self._mcts(state, current_player)
