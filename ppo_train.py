import gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque

# Define the policy network
class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_dim)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return torch.tanh(self.fc3(x))

# Define a replay buffer using deque
class ReplayBuffer:
    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

# Define the PPO agent
class PPOAgent:
    def __init__(self, state_dim, action_dim, lr=0.001, gamma=0.99, epsilon=0.2):
        self.policy = PolicyNetwork(state_dim, action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.gamma = gamma
        self.epsilon = epsilon

    def get_action(self, state):
        state = torch.FloatTensor(state)
        action_mean = self.policy(state)
        dist = torch.distributions.Normal(action_mean, 0.2)  # Adjust std.dev. as needed
        action = dist.sample()
        return action.item(), dist.log_prob(action)

    def update_policy(self, states, actions, advantages, old_probs):
        new_probs = torch.exp(-0.5 * ((actions - self.policy(states)) / 0.2) ** 2) / (0.2 * np.sqrt(2 * np.pi))
        ratio = new_probs / old_probs

        surrogate1 = ratio * advantages
        surrogate2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantages

        loss = -torch.min(surrogate1, surrogate2).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

def compute_advantages(rewards, gamma=0.99, tau=0.95):
    advantages = []
    advantage = 0

    for r in reversed(rewards):
        delta = r + gamma * (1 - tau) * advantage - advantage
        advantage = advantage + delta
        advantages.append(advantage)

    advantages = list(reversed(advantages))
    advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)
    return torch.FloatTensor(advantages)

# Initialize the environment
env = gym.make('Pendulum-v1')
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

# Initialize the PPO agent
ppo_agent = PPOAgent(state_dim, action_dim)
# Initialize the replay buffer
replay_buffer = ReplayBuffer(max_size=10000)

# Training loop
num_episodes = 1000
num_steps = 200  # Max number of steps per episode
num_epochs = 10  # Number of optimization epochs per update
batch_size = 64  # Batch size for mini-batch learning

for episode in range(num_episodes):
    states, actions, rewards, old_probs = [], [], [], []

    for step in range(num_steps):
        state, others = env.reset()
        done = False
        total_reward = 0

        action, old_prob = ppo_agent.get_action(state)
        new_state, reward, done, trunction, _ = env.step([action])

        states.append(state)
        actions.append(action)
        rewards.append(reward)
        old_probs.append(old_prob)

        state = new_state

        if len(states) >= batch_size:
            # Perform a mini-batch policy update
            states_tensor = torch.FloatTensor(np.vstack(states))
            actions_tensor = torch.FloatTensor(actions)
            rewards_tensor = torch.FloatTensor(rewards)
            old_probs_tensor = torch.cat(old_probs)

            advantages = compute_advantages(rewards_tensor)
            ppo_agent.update_policy(states_tensor, actions_tensor, advantages, old_probs_tensor)

            states, actions, rewards, old_probs = [], [], [], []

        # Store the episode in the replay buffer
        episode_experience = (states, actions, rewards, old_probs)
        replay_buffer.add(episode_experience)

    # Sample mini-batches from the replay buffer for training
    for _ in range(num_epochs):
        mini_batch = replay_buffer.sample(batch_size)
        mini_states, mini_actions, mini_rewards, mini_old_probs = zip(*mini_batch)

        # Flatten the list of states and actions
        mini_states = [state for episode_states in mini_states for state in episode_states]
        mini_actions = [action for episode_actions in mini_actions for action in episode_actions]
        mini_rewards = [reward for episode_rewards in mini_rewards for reward in episode_rewards]

        mini_states_tensor = torch.FloatTensor(mini_states)
        mini_actions_tensor = torch.FloatTensor(mini_actions)
        mini_rewards_tensor = torch.FloatTensor(mini_rewards)
        mini_old_probs_tensor = torch.FloatTensor(mini_old_probs)

        advantages = compute_advantages(mini_rewards)
        ppo_agent.update_policy(mini_states_tensor, mini_actions_tensor, advantages, mini_old_probs_tensor)

    print(f"Episode: {episode + 1}")

# Close the environment
env.close()
