# PPO: agent
# Dylan
# 2024.3.4

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
TAU = 0.005
LAMBDA = 0.95
MEMORY_SIZE = 100000
BATCH_SIZE = 64
NUM_EPOCH = 10


# Actor Network
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))  # to initialize the log_std parameter to zero

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        mean = self.mean(x)
        std = self.log_std.exp().expand_as(mean)  # Ensure std is positive and broadcastable with mean
        dist = Normal(mean, std)
        action = dist.sample()  # Sample an action
        log_prob = dist.log_prob(action)  # Compute log probability of the sampled action
        return action, log_prob


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
        self.state_cap = np.empty((self.capacity, state_dim))
        self.action_cap = np.empty((self.capacity, action_dim), dtype=np.float32)
        self.log_prob_cap = torch.empty((self.capacity, action_dim), dtype=torch.float32)
        self.value_cap = torch.empty((self.capacity, 1), dtype=torch.float32)
        self.reward_cap = np.empty((self.capacity, 1))
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
        # Reset the memory (optional method if you need to clear memory at some point)
        self.count = 0
        self.current = 0


# PPO Agent
class PPOAgent:
    def __init__(self, state_dim, action_dim):

        self.epsilon_clip = 0.2

        self.actor = Actor(state_dim, action_dim).to(device)  # move nn to device
        self.actor_target = Actor(state_dim, action_dim).to(device)  # same structure as actor
        self.actor_target.load_state_dict(self.actor.state_dict())  # copy the current nn's weights of actor
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=LR_ACTOR)  # retrieves the parameters

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=LR_CRITIC)

        self.replay_buffer = ReplayMemory(MEMORY_SIZE, state_dim, action_dim, BATCH_SIZE)

    def get_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)  # unsqueeze(0) add a dimension from (3,) to (1,3)
        action = self.actor.forward(state)[0]  # get the action from the actor network
        log_prob = self.actor.forward(state)[1]  # get the log probability of the action
        value = self.critic.forward(state)  # get the value of the state
        # add gaussian noise to the action
        noise = torch.randn_like(action) * 0.1
        action = action + noise
        return action.detach().cpu().numpy()[0], log_prob, value

        # .cpu() is a method that moves a tensor from GPU memory to CPU memory.
        # This is useful if you want to perform operations on the tensor using NumPy on the CPU.

    def update(self):

        for epoch_i in range(NUM_EPOCH):

            states, actions, rewards, log_probs, values, dones = self.replay_buffer.sample()
            states = torch.FloatTensor(states).to(device)
            rewards = torch.FloatTensor(rewards).unsqueeze(1).to(device)
            log_probs = torch.FloatTensor(log_probs).unsqueeze(1).to(device)
            values = torch.FloatTensor(values).to(device)
            dones = torch.FloatTensor(dones).unsqueeze(1).to(device)

            # Calculate GAE
            advantages = torch.empty(len(rewards), dtype=torch.float32)
            for t in range(len(rewards) - 1):
                discount = 1
                a_t = 0
                for k in range(t, len(rewards) - 1):
                    a_t += discount * (rewards[k] + GAMMA * values[k + 1] * (
                            1 - int(dones[k])) - values[k])
                    discount *= GAMMA * LAMBDA
                advantages[t] = a_t
            advantages = torch.FloatTensor(advantages).unsqueeze(1).to(device)
            returns = advantages + values

            # Update critic with returns
            critic_values = self.critic.forward(states)
            critic_loss = F.mse_loss(critic_values, returns)
            self.critic_optimizer.zero_grad()  # zero the gradients
            critic_loss.backward()  # backpropagate the loss
            self.critic_optimizer.step()  # update the weights

            # Actor update using PPO clip
            next_log_probs = self.actor.forward(states)[1]  # Ensure this method returns log probs of actions
            ratio = torch.exp(next_log_probs - log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.epsilon_clip, 1 + self.epsilon_clip) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()

            # # check if related variables are leaf nodes
            # print(f"actor_loss.is_leaf: {actor_loss.is_leaf}")
            # print(f"surr1.is_leaf: {surr1.is_leaf}")
            # print(f"surr2.is_leaf: {surr2.is_leaf}")
            # print(f"advantages.is_leaf: {advantages.is_leaf}")
            # print(f"returns.is_leaf: {returns.is_leaf}")
            # print(f"states.is_leaf: {states.is_leaf}")
            # print(f"log_probs.is_leaf: {log_probs.is_leaf}")
            # print(f"next_log_probs.is_leaf: {next_log_probs.is_leaf}")
            # print(f"values.is_leaf: {values.is_leaf}")

            # states.grad = None  # zero the gradients
            # values.grad = None  # zero the gradients

            self.actor_optimizer.zero_grad()
            actor_loss.backward()  # TODO
            self.actor_optimizer.step()

            # total_loss = actor_loss + 0.5 * critic_loss
            # self.actor_optimizer.zero_grad()
            # self.critic_optimizer.zero_grad()
            # total_loss.backward()
            # self.actor_optimizer.step()
            # self.critic_optimizer.step()

            # Soft-update target networks if they are used in PPO (typically not, but included for completeness)
            for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
                target_param.data.copy_(TAU * param.data + (1.0 - TAU) * target_param.data)

            for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
                target_param.data.copy_(TAU * param.data + (1.0 - TAU) * target_param.data)

        self.replay_buffer.clear_memo()  # Clear the replay buffer after updating the networks
