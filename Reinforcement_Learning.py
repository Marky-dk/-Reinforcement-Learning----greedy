# Bibliotecas utilizadas para o funcionamento do script
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pandas as pd

# Parametros de configuração do modelo
start_state = (5,0) # Ponto inicial
gamma = 0.9 # valor de gamma
n_actions = 4 # número de ações que o agente pode realizar
epsilon = 0.2 # probabilidade de explorar
num_iterations = 150 # Número de tentativar para o agente chegar no objetivo
living_penalty = -1 # punição pelo agente continuar tentando
parede = -2 # punição por bater na parede
chegada = 100 # recompensa por chegar no objetivo
falta = -20 # punição por ir para um lugar que não deveria

# Definindo o ambiente do labirinto:
# 0 -> caminho livre
# 1 -> paredes
# 2 -> falta
# 9 -> objetivos
grid = np.array([
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 9],
    [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    [1, 0, 2, 1, 1, 0, 0, 0, 1, 0, 1],
    [1, 0, 1, 1, 1, 0, 1, 2, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1],
    [0, 0, 1, 1, 1, 0, 2, 1, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1],
    [1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1],
    [1, 2, 1, 2, 1, 0, 0, 0, 0, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
])

# Inicialização dos valores estimados de ação (Q-values) com zeros
q_values = np.zeros((grid.shape[0], grid.shape[1], n_actions))

# Fusão para criar a politica de Epsilon Greedy (fusão responsavel por verificar se o agente irá explorar ou explotar)
def epsilon_greedy_policy(q_values, epsilon, state):
    if np.random.rand() < epsilon:
        # Ação aleatória (exploração)
        return np.random.choice(n_actions)
    else:
        # Ação gananciosa (exploração)
        return np.argmax(q_values[state[0], state[1]])

# Função para mover o agente no ambiente
def move(state, action):
    if action == 0:
        new_state = (state[0] - 1, state[1])  # Mover para cima
    elif action == 1:
        new_state = (state[0] + 1, state[1])  # Mover para baixo
    elif action == 2:
        new_state = (state[0], state[1] - 1)  # Mover para a esquerda
    else:
        new_state = (state[0], state[1] + 1)  # Mover para a direita

    # Verificar se o movimento é válido (dentro do labirinto e não em uma parede)
    if (0 <= new_state[0] < grid.shape[0] and
        0 <= new_state[1] < grid.shape[1] and
        grid[new_state] != 1):
        return new_state
    else:
        return state

# Fusão para criar a Q-table e salvar o valor de q-value para o agente aprender com os erros
def create_q_table(q_values):
    rows, cols, _ = q_values.shape
    q_table = pd.DataFrame(np.zeros((rows, cols), dtype=int), columns=[str(i) for i in range(cols)])

    for row in range(rows):
        for col in range(cols):
            q_table.iloc[row, col] = np.argmax(q_values[row, col])

    return q_table

# Função para atualizar o gráfico da animação
def update(frame):
    ax.cla()
    ax.matshow(grid, cmap=plt.cm.Pastel1)

    if frame < len(path):
        state = path[frame]
        ax.plot(state[1], state[0], 'bo', markersize=15, label="Agente")
        ax.set_title(f"Época {frame + 1}")

# Simulação de tentativas
path = []

q_tables = []  # Lista para armazenar as tabelas Q após cada época

for episode in range(num_iterations):
    state = start_state 
    done = False
    movimento = 0

    while not done:
        action = epsilon_greedy_policy(q_values, epsilon, state)
        new_state = move(state, action)
        movimento += 1

        if grid[new_state] == 9:
            reward = chegada  # Ganhou, encontrou o objetivo
            done = True
        elif grid[new_state] == 1:
            reward = parede  # Bateu em uma parede

        elif grid[new_state] == 2:
            reward = falta  # Perdeu, sofreu falta
            done = True
        else:
            reward = living_penalty  # Outras ações custam um pouco de recompensa negativa



        # Atualizar os valores Q usando a fórmula de aprendizado por reforço Q-learning
        q_values[state[0], state[1], action] += 0.1 * (reward + gamma * np.max(q_values[new_state[0], new_state[1]]) - q_values[state[0], state[1], action])

        print(f"Tentativas: {episode + 1}:")
        print(f"Estado anterior: {state}")
        print(f"Ação tomada pelo agente: {action}")
        print(f"Estado atual: {new_state}")
        print(f"Recompensa: {reward}")
        print(f"Movimento: {movimento}\n")

        state = new_state
        path.append(state)  # Adiciona o estado à lista de posições do agente

        q_tables.append(create_q_table(q_values))  # Adiciona a tabela Q atual à lista

# Preparando a representação gráfica do labirinto
fig, ax = plt.subplots()

# Configurações de plotagem
ax.set_xticks([])
ax.set_yticks([])

# Criando a animação
ani = FuncAnimation(fig, update, frames=len(path), repeat=False)
plt.show()

print("Log da Q-table\n")
print(q_tables)

