# PPO: agent
# Dylan
# 2024.3.16

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device: ", device)

LR_ACTOR = 1e-3
LR_CRITIC = 1e-3
GAMMA = 0.99
TAU = 0.01
LAMBDA = 0.95
MEMORY_SIZE = 100000


# Actor Network
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean = nn.Linear(hidden_dim, action_dim)
        # Change log_std to be action-dependent
        self.log_std = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        mean = self.mean(x)
        log_std = self.log_std(x)
        std = log_std.exp() # Ensure std is positive and allow different std for each action
        normal_dist = Normal(mean, std)

        z = normal_dist.rsample()  # For reparameterization trick
        action = torch.tanh(z)  # Maps to (-1, 1)
        # print(f"action: {action}")
        action_log_prob = normal_dist.log_prob(z) - torch.log(1 - action.pow(2) + 1e-6)
        action_log_prob = action_log_prob.sum(axis=-1, keepdim=True)  # Correction for the tanh squashing

        low, high = -2, 2
        action = low + (action - low) * (high - low) / (high - low)  # Scale and shift to [-2,2]
        # action = action.clamp(low, high)  # Ensure action is within bounds

        return action, action_log_prob


# Critic Network
class Critic(nn.Module):
    def __init__(self, state_dim, hidden_dim=64):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        v = self.fc3(x)
        return v


# Replay Buffer
class ReplayMemory:
    def __init__(self, capacity, state_dim, action_dim, batch_size):
        self.capacity = capacity
        self.batch_size = batch_size
        self.state_cap = np.empty((self.capacity, state_dim), dtype=np.float32)
        self.action_cap = np.empty((self.capacity, action_dim), dtype=np.float32)
        self.log_prob_cap = np.empty((self.capacity, action_dim), dtype=np.float32)
        self.value_cap = np.empty((self.capacity, 1), dtype=np.float32)
        self.reward_cap = np.empty((self.capacity, 1), dtype=np.float32)
        self.done_cap = np.empty((self.capacity, 1), dtype=bool)

        self.count = 0  # To keep track of the actual number of experiences stored
        self.current = 0  # To keep track of the current position for storing the next experience

    def add_memo(self, state, action, reward, log_prob, value, done):
        self.state_cap[self.count] = state
        self.action_cap[self.count] = action
        self.log_prob_cap[self.count] = log_prob
        self.value_cap[self.count] = value
        self.reward_cap[self.count] = reward
        self.done_cap[self.count] = done
        self.count = min(self.count + 1, self.capacity)  # Ensure count does not exceed capacity
        self.current += 1  # Move to the next position

    def sample(self):
        # Ensure we can only sample if we have enough experiences
        max_samples = min(self.count, self.batch_size)
        idxes = np.random.choice(range(self.count), max_samples, replace=False)

        states = self.state_cap[idxes]
        actions = self.action_cap[idxes]
        log_probs = self.log_prob_cap[idxes]
        values = self.value_cap[idxes]
        rewards = self.reward_cap[idxes]
        dones = self.done_cap[idxes]

        return states, actions, log_probs, values, rewards, dones

    def clear_memo(self):
        self.count = 0
        self.current = 0


# PPO Agent
class PPOAgent:
    def __init__(self, state_dim, action_dim, batch_size):

        self.epsilon_clip = 0.2
        self.batch_size = batch_size

        self.actor = Actor(state_dim, action_dim).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=LR_ACTOR)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=LR_CRITIC)

        self.replay_buffer = ReplayMemory(MEMORY_SIZE, state_dim, action_dim, self.batch_size)

    def get_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        action, log_prob = self.actor.forward(state)
        value = self.critic.forward(state)  # get the value of the state
        # noise = torch.randn_like(action) * 0.1  # TODO: add gaussian noise to the action
        # action = action + noise
        return action.detach().cpu().numpy()[0], log_prob.detach().cpu().numpy()[0], value.detach().cpu().numpy()[0]

    def update(self):

        states, actions, rewards, log_probs, values, dones = self.replay_buffer.sample()
        states = torch.FloatTensor(states).to(device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(device)
        log_probs = torch.FloatTensor(log_probs).unsqueeze(1).to(device)
        values = torch.FloatTensor(values).to(device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(device)
        # print(f"rewards: {rewards}")

        advantages = torch.empty(len(rewards), dtype=torch.float32)
        for t in range(len(rewards) - 1):
            discount = 1
            a_t = 0
            for k in range(t, len(rewards) - 1):
                a_t += discount * (rewards[k] + GAMMA * values[k + 1] * (1 - int(dones[k])) - values[k])
                discount *= GAMMA * LAMBDA
            advantages[t] = a_t
        advantages = torch.FloatTensor(advantages).unsqueeze(1).to(device)

        # Actor update using PPO clip
        next_log_probs = self.actor.forward(states)[1]
        ratio = torch.exp(next_log_probs - log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.epsilon_clip, 1 + self.epsilon_clip) * advantages
        actor_loss = -torch.min(surr1, surr2).mean()
        print(f"actor loss: {round(actor_loss,2)}")


        # Update critic with returns
        returns = advantages + values
        critic_values = self.critic.forward(states)
        critic_loss = F.mse_loss(critic_values, returns)
        print(f"critic loss: {round(critic_loss,2)}")

        # Method 1: separate actor and critic loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        self.critic_optimizer.zero_grad()  # zero the gradients
        critic_loss.backward()
        self.critic_optimizer.step()  # update the weights

        # Method 2: combined total loss
        # total_loss = actor_loss + 0.1 *critic_loss
        # self.actor_optimizer.zero_grad()
        # self.critic_optimizer.zero_grad()
        # total_loss.backward()
        # self.actor_optimizer.step()
        # self.critic_optimizer.step()


        # self.replay_buffer.clear_memo()  # Clear the replay buffer after updating the networks
