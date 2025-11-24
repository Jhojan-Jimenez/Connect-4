# Magnus - Agente Connect-4 con MCTS y Q-Learning

Implementaci�n de un agente inteligente para Connect-4 que combina **Monte Carlo Tree Search (MCTS)** con **Q-Learning** para mejorar su rendimiento mediante aprendizaje por refuerzo.

---

## Presentaci�n

**Link a las diapositivas**: [Ver presentaci�n en Canva](https://www.canva.com/design/DAG5kSSWAdQ/lw8idiZvdCJ73S68OichHA/edit?utm_content=DAG5kSSWAdQ&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton)

---

## Par�metros de Configuraci�n

| Par�metro              | Descripci�n                                              | Valor por defecto |
| ---------------------- | -------------------------------------------------------- | ----------------- |
| `simulations`          | N�mero de simulaciones MCTS por jugada                   | `250`             |
| `exploration_c`        | Par�metro C de UCB1 para balance exploraci�n-explotaci�n | `2`               |
| `alpha`                | Tasa de aprendizaje para actualizaci�n de Q-values       | `0.3`             |
| `beta`                 | Peso de mezcla entre Prior (Q-table) y UCB1              | `0.7`             |
| `confidence_threshold` | Visitas necesarias para considerar Q-value confiable     | `50`              |
| `q_file`               | Archivo con valores Q precargados                        | `"magnus_q.pkl"`  |

---

## Autores

Jhojan Camilo Jimenez
Daniel Ramirez Chinchilla
