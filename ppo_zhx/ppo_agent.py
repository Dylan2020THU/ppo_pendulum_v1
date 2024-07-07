# PPO: agent
# Dylan
# 2024.3.16
# 2024.7.6

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device: ", device)


# Actor Network
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_mean = nn.Linear(hidden_dim, action_dim)
        self.fc_std = nn.Linear(hidden_dim, action_dim)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.softplus = nn.Softplus()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        mean = self.tanh(self.fc_mean(x)) * 2
        std = self.softplus(self.fc_std(x)) + 1e-3
        normal_dist = Normal(mean, std)

        action = normal_dist.sample()
        action = action.clamp(-2.0, 2.0)
        action_log_prob = normal_dist.log_prob(action)
        entropy = normal_dist.entropy().mean()

        return action, action_log_prob, entropy


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
    def __init__(self, batch_size):
        self.BATCH_SIZE = batch_size
        self.state_cap = []
        self.log_prob_cap = []
        self.value_cap = []
        self.action_cap = []
        self.reward_cap = []
        self.done_cap = []

    def add_memo(self, state, action, reward, log_prob, value, done):
        self.state_cap.append(state)
        self.action_cap.append(action)
        self.reward_cap.append(reward)
        self.log_prob_cap.append(log_prob)
        self.value_cap.append(value)
        self.done_cap.append(done)

    def sample(self):
        num_state = len(self.state_cap)
        batch_start_points = np.arange(0, num_state, self.BATCH_SIZE)
        memory_indices = np.arange(num_state, dtype=np.int32)
        np.random.shuffle(memory_indices)
        batches = [memory_indices[i:i + self.BATCH_SIZE] for i in batch_start_points]

        return np.array(self.state_cap), \
               np.array(self.action_cap), \
               np.array(self.reward_cap), \
               np.array(self.log_prob_cap), \
               np.array(self.done_cap), \
               batches

    def clear_memo(self):
        self.state_cap = []
        self.action_cap = []
        self.reward_cap = []
        self.log_prob_cap = []
        self.done_cap = []


# PPO Agent
class PPOAgent:
    def __init__(self, state_dim, action_dim):
        self.LR_ACTOR = 3e-4
        self.LR_CRITIC = 3e-4
        self.GAMMA = 0.99
        self.LAMBDA = 0.95
        self.EPSILON_CLIP = 0.2
        self.EPOCH = 10
        self.BATCH_SIZE = 64
        self.ENTROPY_COEFF = 0.01

        self.actor = Actor(state_dim, action_dim).to(device)
        self.critic = Critic(state_dim).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.LR_ACTOR)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.LR_CRITIC)
        self.replay_buffer = ReplayMemory(self.BATCH_SIZE)

    def get_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        action, log_prob, entropy = self.actor.forward(state)
        value = self.critic.forward(state)
        return action.detach().cpu().numpy()[0], log_prob.detach().cpu().numpy()[0], value.detach().cpu().numpy()[
            0], entropy.item()

    def update(self):
        for epoch_i in range(self.EPOCH):
            memo_states, memo_actions, memo_rewards, memo_action_log_probs, memo_dones, memo_batches = self.replay_buffer.sample()

            memo_values = self.critic(torch.FloatTensor(memo_states).to(device)).detach().cpu().numpy()
            advantages = np.zeros(len(memo_rewards), dtype=np.float32)
            gae = 0
            for t in reversed(range(len(memo_rewards) - 1)):
                delta = memo_rewards[t] + self.GAMMA * memo_values[t + 1] * (1 - memo_dones[t]) - memo_values[t]
                # tmp1 = memo_rewards[t]
                # tmp2 = self.GAMMA * memo_values[t + 1] * (1 - memo_dones[t]) - memo_values[t]
                # delta = tmp1 + tmp2
                gae = delta + self.GAMMA * self.LAMBDA * gae
                advantages[t] = gae

            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            advantages_tensor = torch.tensor(advantages, dtype=torch.float32).to(device)

            for batch_i in memo_batches:
                batch_states_tensor = torch.FloatTensor(memo_states[batch_i]).to(device)
                batch_actions_tensor = torch.FloatTensor(memo_actions[batch_i]).to(device)
                batch_old_log_probs_tensor = torch.FloatTensor(memo_action_log_probs[batch_i]).to(device)

                _, new_log_probs_tensor, entropy_tensor = self.actor.forward(batch_states_tensor)

                ratio = torch.exp(new_log_probs_tensor - batch_old_log_probs_tensor)
                surr1 = ratio * advantages_tensor[batch_i]
                surr2 = torch.clamp(ratio, 1 - self.EPSILON_CLIP, 1 + self.EPSILON_CLIP) * advantages_tensor[batch_i]
                actor_loss = -torch.min(surr1, surr2).mean() - self.ENTROPY_COEFF * entropy_tensor.mean()

                returns = advantages_tensor[batch_i] + torch.FloatTensor(memo_values[batch_i]).to(device)
                critic_loss = nn.MSELoss()(self.critic(batch_states_tensor), returns)

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                self.critic_optimizer.step()

        self.replay_buffer.clear_memo()
