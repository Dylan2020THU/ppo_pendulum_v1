# PPO: training
# Dylan
# 2024.3.16

import gym
import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import time
from ppo_agent import PPOAgent

# Initialize environment
scenario = "Pendulum-v1"
env = gym.make(id=scenario)
STATE_DIM = env.observation_space.shape[0]
ACTION_DIM = env.action_space.shape[0]

# Hyperparameters
NUM_EPISODE = 300
NUM_STEP = 200
UPDATE_INTERVAL = 20

# Initialize agent
agent = PPOAgent(STATE_DIM, ACTION_DIM)

# Training Loop
REWARD_BUFFER = np.empty(shape=NUM_EPISODE)
# total_step = 0
for episode_i in range(NUM_EPISODE):
    state, others = env.reset()  # state: ndarray, others: dict
    done = False
    episode_reward = 0

    for step_i in range(NUM_STEP):
        # while not done:
        total_step = episode_i * NUM_STEP + step_i + 1
        # total_step += 1
        # print(f"total_step: {total_step}")
        action, log_prob, value = agent.get_action(state)
        next_state, reward, done, truncation, info = env.step(action)
        # print(f"{done} at step {total_step}")
        agent.replay_buffer.add_memo(state, action, reward, log_prob, value, done)

        # if step_i == NUM_STEP - 1:
        #     done = True
        if total_step % UPDATE_INTERVAL == 0:
            agent.update()
        state = next_state
        episode_reward += reward

        if done:
            print(f"{done} at step {total_step}.")
            break

    REWARD_BUFFER[episode_i] = episode_reward

    print(f"Episode: {episode_i + 1}, Reward: {round(episode_reward, 2)}")

env.close()

current_path = os.path.dirname(os.path.realpath(__file__))
agent_path = current_path + "/models/" + scenario + '/'
timestamp = time.strftime("%Y%m%d%H%M%S")

# Save models
flag = os.path.exists(agent_path)
if not flag:
    os.makedirs(agent_path)
actor_name = agent_path + f'ppo_actor_{timestamp}.pth'
torch.save(agent.actor.state_dict(), actor_name)

# Save the rewards as txt file
np.savetxt(current_path + f'/ppo_reward_{timestamp}.txt', REWARD_BUFFER)

# Plot rewards using ax.plot()
plt.plot(REWARD_BUFFER, color='purple', label='Episode Reward')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.grid()
plt.title(scenario)
plt.legend()
plt.savefig(f"reward_{timestamp}.png", format='png')
plt.show()
