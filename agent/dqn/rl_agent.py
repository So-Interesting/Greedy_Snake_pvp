# -*- coding:utf-8  -*-
# Time  : 2021/5/27 下午3:38
# Author: Yahui Cui

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random

def get_surrounding(state, width, height, x, y):
    surrounding = [state[(y - 1) % height][x],  # up
                   state[(y + 1) % height][x],  # down
                   state[y][(x - 1) % width],  # left
                   state[y][(x + 1) % width]]  # right

    return surrounding

def get_surrounding_3(state, width, height, x, y):
    surrounding = [state[(y - 1) % height][(x - 1) % width], 
                   state[(y - 1) % height][x],
                   state[(y - 1) % height][(x + 1) % width], 
                   state[(y + 1) % height][(x - 1) % width], 
                   state[(y + 1) % height][x],
                   state[(y + 1) % height][(x + 1) % width],  
                   state[y][(x - 1) % width],  
                   state[y][(x + 1) % width],
                   state[y][(x - 2) % width],
                   state[y][(x + 2) % width],
                   state[(y - 2) % height][x],
                   state[(y + 2) % height][x]]

    return surrounding

# Self position:        0:head_x; 1:head_y
# Head surroundings:    2:head_up; 3:head_down; 4:head_left; 5:head_right
# Beans positions:      (6, 7) (8, 9) (10, 11) (12, 13) (14, 15)
# Other snake positions: (16, 17) -- (other_x - self_x, other_y - self_y)
def get_observations(state, info, agents_index, obs_dim, height, width, step):
    state = np.array(state)
    state = np.squeeze(state, axis=2)
    observations = np.zeros((len(agents_index), obs_dim))
    snakes_position = np.array(info['snakes_position'], dtype=object)
    beans_position = np.array(info['beans_position']).flatten()
    for i, j in enumerate(agents_index):
        # self head position
        observations[i][:2] = snakes_position[j][0][:]

        # head surroundings
        head_x = snakes_position[i][0][1]
        head_y = snakes_position[i][0][0]
        head_surrounding = get_surrounding_3(state, width, height, head_x, head_y)
        observations[i][2:14] = head_surrounding[:]

        head_x_U = snakes_position[i ^ 1][0][1]
        head_y_U = snakes_position[i ^ 1][0][0]
        head_surrounding = get_surrounding_3(state, width, height, head_x_U, head_y_U)
        observations[i][14:26] = head_surrounding[:]
        # other snake positions
        snake_heads = [snake[0] for snake in snakes_position]
        snake_heads = np.array(snake_heads[1:])
        snake_heads -= snakes_position[i][0]
        observations[i][26:28] = snake_heads.flatten()[:]
        observations[i][28:30] = [len(snake) for snake in snakes_position]

        observations[i][30] = step
        # beans positions
        beans_len = len(beans_position)
        observations[i][31 : 31 + beans_len] = beans_position[:]
        if (beans_len < 10) : observations[31 + beans_len:] = 0
    return observations


class Critic(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x


class DQN(object):
    def __init__(self, state_dim, action_dim, num_agent, hidden_size):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_agent = num_agent

        self.hidden_size = hidden_size

        self.critic_eval = Critic(self.state_dim, self.action_dim, self.hidden_size)
        self.critic_target = Critic(self.state_dim, self.action_dim, self.hidden_size)

    def choose_action(self, observation):
        observation = torch.tensor(observation, dtype=torch.float).view(1, -1)
        action = torch.argmax(self.critic_eval(observation)).item()
        return action

    def load(self, file):
        base_path = os.path.dirname(os.path.abspath(__file__))
        file = os.path.join(base_path, file)
        self.critic_eval.load_state_dict(torch.load(file))
        self.critic_target.load_state_dict(torch.load(file))
    
    def store_transition(self, transition):
        if len(self.buffer) == self.buffer_size:
            self.buffer.pop(0)
        self.buffer.append(transition)

    def learn(self):
        if len(self.buffer) < self.batch_size:
            return

        samples = random.sample(self.buffer, self.batch_size)
        obs, action, reward, obs_, done = zip(*samples)

        obs = torch.tensor(obs, dtype=torch.float).squeeze()
        action = torch.tensor(action, dtype=torch.long).view(self.batch_size, -1)
        reward = torch.tensor(reward, dtype=torch.float).view(self.batch_size, -1).squeeze()
        obs_ = torch.tensor(obs_, dtype=torch.float).squeeze()
        done = torch.tensor(done, dtype=torch.float).view(self.batch_size, -1).squeeze()

        q_eval = self.critic_eval(obs).gather(1, action)
        q_next = self.critic_target(obs_).detach()
        q_target = (reward + self.gamma * q_next.max(1)[0] * (1 - done)).view(self.batch_size, 1)
        loss_fn = nn.MSELoss()
        loss = loss_fn(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.learn_step_counter % self.target_replace_iter == 0:
            self.learn_step_counter = 0
            self.critic_target.load_state_dict(self.critic_eval.state_dict())
        self.learn_step_counter += 1

        self.loss = loss.item()

        return loss

    def save(self, run_dir, episode):
        base_path = os.path.join(run_dir, 'trained_model')
        if not os.path.exists(base_path):
            os.makedirs(base_path)

        model_critic_path = os.path.join(base_path, "critic_" + str(episode) + ".pth")
        torch.save(self.critic_eval.state_dict(), model_critic_path)


def to_joint_action(actions, num_agent):
    joint_action = []
    for i in range(num_agent):
        action = actions
        one_hot_action = [0] * 4
        one_hot_action[action] = 1
        one_hot_action = [one_hot_action]
        joint_action.append(one_hot_action)
    return joint_action

agent = DQN(41, 4, 1, 256)
agent.load('critic_5000.pth')

def my_controller(observation_list, a, b):
    obs_dim = 41
    obs = observation_list[0]
    board_width = obs['board_width']
    board_height = obs['board_height']
    ctrl_agent_index = [obs['controlled_snake_index'] for obs in observation_list]
    state_map = obs['state_map']
    info = {"beans_position": obs[1], "snakes_position": [obs[key] for key in obs.keys() & {2, 3}]}

    observations = get_observations(state_map, info, ctrl_agent_index, obs_dim, board_height, board_width)
    actions = agent.choose_action(observations)

    return to_joint_action(actions, len(ctrl_agent_index))