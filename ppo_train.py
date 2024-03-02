# PPO: training
# Dylan
# 2024.3.2

import gym
import numpy as np
import torch
# import matplotlib.pyplot as plt
import os
import time
from ppo_agent import DDPGAgent
import random

# Initialize environment
env = gym.make(id='Pendulum-v1')
STATE_DIM = env.observation_space.shape[0]
ACTION_DIM = env.action_space.shape[0]

# Hyperparameters
NUM_EPISODE = 100
NUM_STEP = 200

# Initialize agent
agent = DDPGAgent(STATE_DIM, ACTION_DIM)

# Training Loop
REWARD_BUFFER = np.empty(shape=NUM_EPISODE)
for episode_i in range(NUM_EPISODE):
    state, others = env.reset()  # state: ndarray, others: dict
    episode_reward = 0

    for step_i in range(NUM_STEP):
        # Select action
        action = agent.get_action(state)
        # Execute action at and observe reward rt and observe new state st+1
        next_state, reward, done, truncation, info = env.step(action)
        # Store transition (st; at; rt; st+1) in R
        agent.replay_buffer.add_memo(state, action, reward, next_state, done)

        state = next_state
        episode_reward += reward

        agent.update()
        if done:
            break

    REWARD_BUFFER[episode_i] = episode_reward

    print(f"Episode: {episode_i + 1}, Reward: {round(episode_reward, 2)}")

current_path = os.path.dirname(os.path.realpath(__file__))
model = current_path + '/models/'
timestamp = time.strftime("%Y%m%d%H%M%S")

# Save models
actor_name = model + f'ddpg_actor_{timestamp}.pth'
critic_name = model + f'ddpg_critic_{timestamp}.pth'
torch.save(agent.actor.state_dict(), actor_name)
torch.save(agent.critic.state_dict(), critic_name)

# Close environment
env.close()

# Save the rewards as txt file
np.savetxt(current_path + f'/ddpg_reward_{timestamp}.txt', REWARD_BUFFER)

# # Plot rewards using ax.plot()
# plt.plot(REWARD_BUFFER)
# plt.xlabel('Episode')
# plt.ylabel('Reward')
# plt.title('DDPG Reward')
# plt.grid()
# plt.show()
# plt.savefig(current_path + f'/ddpg_reward_{timestamp}.png')
