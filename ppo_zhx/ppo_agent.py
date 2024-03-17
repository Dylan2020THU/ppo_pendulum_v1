# PPO: agent
# Dylan
# 2024.3.17

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
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
        self.mean = nn.Linear(hidden_dim, action_dim)
        # Change log_std to be action-dependent
        self.log_std = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        mean = self.mean(x)
        log_std = self.log_std(x)
        std = log_std.exp()  # Ensure std is positive and allow different std for each action
        normal_dist = Normal(mean, std)

        z = normal_dist.rsample()  # For reparameterization trick
        action = torch.tanh(z)  # Maps to (-1, 1)
        # print(f"action: {action}")
        action_log_prob = normal_dist.log_prob(z) - torch.log(1 - action.pow(2) + 1e-6)
        action_log_prob = action_log_prob.sum(axis=-1, keepdim=True)  # Correction for the tanh squashing

        low, high = -2, 2
        action = low + 0.5 * (action + 1.0) * (high - low)  # Scale and shift to [-2,2]
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
        # print(f"num_state: {num_state}")
        batch_start_points = np.arange(0, num_state, self.BATCH_SIZE)
        memory_indices = np.arange(num_state, dtype=np.int32)
        np.random.shuffle(memory_indices)
        batches = [memory_indices[i:i + self.BATCH_SIZE] for i in batch_start_points]

        return np.array(self.state_cap), \
            np.array(self.action_cap), \
            np.array(self.reward_cap), \
            np.array(self.log_prob_cap), \
            np.array(self.value_cap), \
            np.array(self.done_cap), \
            batches

    def clear_memo(self):
        self.state_cap = []
        self.action_cap = []
        self.reward_cap = []
        self.log_prob_cap = []
        self.value_cap = []
        self.done_cap = []


# PPO Agent
class PPOAgent:
    def __init__(self, state_dim, action_dim):

        self.LR_ACTOR = 1e-3
        self.LR_CRITIC = 1e-3
        self.GAMMA = 0.99
        self.LAMBDA = 0.95
        self.TAU = 0.005
        self.EPSILON_CLIP = 0.2
        self.EPOCH = 10
        self.BATCH_SIZE = 5

        self.actor = Actor(state_dim, action_dim).to(device)
        self.target_actor = Actor(state_dim, action_dim).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.LR_ACTOR)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.target_critic = Critic(state_dim, action_dim).to(device)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.LR_CRITIC)

        self.replay_buffer = ReplayMemory(self.BATCH_SIZE)

    def get_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        action, log_prob = self.actor.forward(state)
        value = self.critic.forward(state)  # get the value of the state
        # noise = torch.randn_like(action) * 0.1  # TODO: add gaussian noise to the action
        # action = action + noise
        return action.detach().cpu().numpy()[0], log_prob.detach().cpu().numpy()[0], value.detach().cpu().numpy()[0]

    def update(self):

        for epoch_i in range(self.EPOCH):
            memo_states, memo_actions, memo_rewards, memo_action_log_probs, memo_values, memo_dones, \
                memo_batches = self.replay_buffer.sample()
            # print(f"rewards: {rewards}")

            memo_values = memo_values.squeeze(1)
            advantages = np.empty(len(memo_rewards), dtype=np.float32)
            for t in range(len(memo_rewards) - 1):
                discount = 1
                a_t = 0
                for k in range(t, len(memo_rewards) - 1):
                    a_t += discount * (memo_rewards[k] + self.GAMMA * memo_values[k + 1] * (1 - int(memo_dones[k])) - memo_values[k])
                    discount *= self.GAMMA * self.LAMBDA
                advantages[t] = a_t

            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            advantages_tensor = torch.tensor(advantages, dtype=torch.float32).to(device)
            memo_values_tensor = torch.tensor(memo_values, dtype=torch.float32).to(device)

            # N actors work simultaneously
            for batch_i in memo_batches:
                memo_batch_states_tensor = torch.tensor(memo_states[batch_i]).to(device)
                memo_batch_action_log_probs_tensor = torch.tensor(memo_action_log_probs[batch_i]).to(device)
                # memo_actions_tensor = torch.tensor(memo_actions[batch_i]).to(device)

                # Actor loss
                next_action_log_probs_tensor = self.actor.forward(memo_batch_states_tensor)[1]
                ratio = torch.exp(next_action_log_probs_tensor - memo_batch_action_log_probs_tensor)
                surr1 = ratio * advantages_tensor[batch_i]
                surr2 = torch.clamp(ratio, 1 - self.EPSILON_CLIP, 1 + self.EPSILON_CLIP) * advantages_tensor[batch_i]
                actor_loss = -torch.min(surr1, surr2).mean()
                # print(f"actor loss: {actor_loss}")

                # Critic loss
                returns = advantages_tensor[batch_i] + memo_values_tensor[batch_i]
                critic_values = self.critic.forward(memo_batch_states_tensor).squeeze(1)
                critic_loss = nn.MSELoss()(critic_values, returns)
                # print(f"critic loss: {critic_loss}")

                # Method 1: separate actor and critic loss
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()
                self.critic_optimizer.zero_grad()  # zero the gradients
                critic_loss.backward()
                self.critic_optimizer.step()  # update the weights

                # # Method 2: combined total loss
                # total_loss = actor_loss + 0.2 *critic_loss
                # self.actor_optimizer.zero_grad()
                # self.critic_optimizer.zero_grad()
                # total_loss.backward()
                # self.actor_optimizer.step()
                # self.critic_optimizer.step()

                # # Update target critic
                # for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
                #     target_param.data.copy_(self.TAU * param.data + (1.0 - self.TAU) * target_param.data)
                #
                # # Update target actor
                # for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
                #     target_param.data.copy_(self.TAU * param.data + (1.0 - self.TAU) * target_param.data)

        self.replay_buffer.clear_memo()  # Clear the replay buffer after updating the networks
