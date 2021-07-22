# -*- coding:utf-8  -*-
# Time  : 2021/5/27 下午3:38
# Author: Yahui Cui

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import math

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


def diji(state, X, Y, width, height):
    mp=np.zeros((height,width))
    for i in range(height):
        for j in range(width):
            mp[i][j]=math.inf
    mp[X][Y]=0
    vis=np.zeros((height,width))
    from queue import PriorityQueue as PQ
    pq=PQ()
    pq.put((0,(X,Y)))
    dx = [-1,1,0,0]
    dy = [0,0,-1,1]
    while (not pq.empty()):
        (d, (x,y)) =pq.get()
        if (vis[x][y]==1): continue
        vis[x][y] = 1
        for i in range(4):
            x1=x+dx[i]
            y1=y+dy[i]
            x1 += height
            x1 %= height
            y1 += width
            y1 %= width
            if (state[x1][y1]==2 or state[x1][y1]==3): continue
            if (mp[x1][y1]>mp[x][y]+1):
                mp[x1][y1]=mp[x][y]+1
                pq.put((mp[x1][y1],(x1,y1)))
    return mp

def get_min_bean(x, y, beans_position, width, height, snakes, state):
    min_distance = math.inf
    min_x = beans_position[0][1]
    min_y = beans_position[0][0]
    index = 0
    Ux = snakes[0][0][1]
    Uy = snakes[0][0][0]
    id = 1
    if (Ux== x and Uy==y):
        Ux = snakes[1][0][1]
        Uy = snakes[1][0][0]
        id = 0
    mat = diji(state,y,x,width, height)
    matU= diji(state,Uy, Ux, width,height)
    for i, (bean_y, bean_x) in enumerate(beans_position):
        # distance = math.sqrt((x - bean_x) ** 2 + (y - bean_y) ** 2)
        distance_my = mat[bean_y][bean_x]
        distance_U = matU[bean_y][bean_x]
        if (len(snakes[id])+1<=len(snakes[id^1])):
            distance = distance_my
        else:
            if (distance_U == math.inf and distance_my == math.inf):
                distance = math.inf
            elif (distance_my == math.inf):
                    distance = math.inf
            elif (distance_U == math.inf):
                distance = distance_my *0.6
            elif (distance_U == distance_my == 1):
                distance = math.inf
            else:
                distance = 0.9*distance_my-0.1*distance_U
        # snake_id = get_id(y, x, width)
        # beans_id = get_id(bean_y, bean_x, width)
        # distance = mat[snake_id][beans_id]
        if distance < min_distance:
            min_x = bean_x
            min_y = bean_y
            min_distance = distance
            index = i
    return min_x, min_y, index

# Self position:        0:head_x; 1:head_y
# Head surroundings:    2:head_up; 3:head_down; 4:head_left; 5:head_right
# Beans positions:      (6, 7) (8, 9) (10, 11) (12, 13) (14, 15)
# Other snake positions: (16, 17) -- (other_x - self_x, other_y - self_y)
def get_observations(state, info, agents_index, obs_dim, height, width, step):
    state = np.array(state)
    state = np.squeeze(state, axis=2)
    observations = np.zeros((len(agents_index), obs_dim))
    snakes_position = np.array(info['snakes_position'], dtype=object)
    # beans_position = np.array(info['beans_position']).flatten()
    beans_position = np.array(info['beans_position'])
    for i in agents_index:
        # self head position
        observations[i][:2] = snakes_position[i][0][:]

        # head surroundings
        head_x = snakes_position[i][0][1]
        head_y = snakes_position[i][0][0]
        head_surrounding = get_surrounding_3(state, width, height, head_x, head_y)
        observations[i][2:14] = head_surrounding[:]
        # observations[i][14:16] = [head_x, head_y]
        observations[i][14:16] = [snakes_position[i][1][1],  snakes_position[i][1][0]]
        observations[i][16:18] = [snakes_position[i][-1][1],  snakes_position[i][-1][0]]

        head_x_U = snakes_position[i ^ 1][0][1]
        head_y_U = snakes_position[i ^ 1][0][0]
        head_surrounding = get_surrounding_3(state, width, height, head_x_U, head_y_U)
        observations[i][18:30] = head_surrounding[:]
        # observations[i][32:34] = [head_x_U, head_y_U]
        observations[i][30:32] = [snakes_position[i^1][1][1],  snakes_position[i^1][1][0]]
        observations[i][32:34] = [snakes_position[i^1][-1][1],  snakes_position[i^1][-1][0]]
        # other snake positions
        snake_heads = [snake[0] for snake in snakes_position]
        snake_heads = np.array(snake_heads[1:])
        snake_heads -= snakes_position[i][0]
        observations[i][34:36] = snake_heads.flatten()[:]
        observations[i][36:38] = [len(snake) for snake in snakes_position]

        observations[i][38] = step
        # beans positions
        beans = beans_position.flatten()
        beans_len = len(beans)
        observations[i][39 : 39 + beans_len] = beans[:]
        if (beans_len < 10) : observations[i][39 + beans_len:] = 0
        for j, (beans_y, beans_x) in enumerate(beans_position):
            dis_my = min(abs(head_y - beans_y), abs(head_y + beans_y + 2 - height)) + min(abs(head_x - beans_x), abs(head_x + beans_x + 2 - width))
            dis_U = min(abs(head_y_U - beans_y), abs(head_y_U + beans_y + 2 - height)) + min(abs(head_x_U - beans_x), abs(head_x_U + beans_x + 2 - width))
            observations[i][49 + 2 * j - 2] = dis_my
            observations[i][49 + 2 * j - 1] = dis_U
        if (beans_len < 10) : observations[i][49 + beans_len:] = 0
        observations[i][59:62] = get_min_bean(head_x, head_y, beans_position, width, height, snakes_position, state)
        observations[i][62:65] = get_min_bean(head_x_U, head_y_U, beans_position, width, height, snakes_position, state)
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

agent = DQN(65, 4, 1, 256)
agent.load('critic_5000.pth')

def my_controller(observation_list, a, b):
    obs_dim = 65
    obs = observation_list[0]
    board_width = obs['board_width']
    board_height = obs['board_height']
    ctrl_agent_index = [obs['controlled_snake_index'] for obs in observation_list]
    state_map = obs['state_map']
    info = {"beans_position": obs[1], "snakes_position": [obs[key] for key in obs.keys() & {2, 3}]}

    observations = get_observations(state_map, info, ctrl_agent_index, obs_dim, board_height, board_width)
    actions = agent.choose_action(observations)

    return to_joint_action(actions, len(ctrl_agent_index))