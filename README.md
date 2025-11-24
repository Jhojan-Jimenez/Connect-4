# Magnus Carlsen

Implementación de un agente inteligente para Connect-4 que combina **Monte Carlo Tree Search (MCTS)** con **Q-Learning** para mejorar su rendimiento mediante entrenamiento.

---

## Presentación

**Link a las diapositivas**: [Ver presentación en Canva](https://www.canva.com/design/DAG5kSSWAdQ/lw8idiZvdCJ73S68OichHA/edit?utm_content=DAG5kSSWAdQ&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton)

---

## Parámetros de Configuración

| Parámetro              | Descripción                                              | Valor por defecto |
| ---------------------- | -------------------------------------------------------- | ----------------- |
| `simulations`          | Número de simulaciones MCTS por jugada                   | `250`             |
| `exploration_c`        | Parámetro C de UCB1 para balance exploración-explotación | `2`               |
| `alpha`                | Tasa de aprendizaje para actualización de Q-values       | `0.3`             |
| `beta`                 | Peso de mezcla entre Prior (Q-table) y UCB1              | `0.7`             |
| `confidence_threshold` | Visitas necesarias para considerar Q-value confiable     | `50`              |
| `q_file`               | Archivo con valores Q precargados                        | `"magnus_q.pkl"`  |

---

## Autores

Jhojan Camilo Jimenez
Daniel Ramirez Chinchilla
