import numpy as np
import torch
import torch.nn as nn

from typing import Union
from torch.distributions import Categorical

from pathlib import Path
import sys
base_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(base_dir))
from agent.greedy.greedy_agent import greedy_snake
from types import SimpleNamespace as SN
import yaml
import math
import os

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def hard_update(source, target):
    target.load_state_dict(source.state_dict())


def soft_update(source, target, tau):
    for src_param, tgt_param in zip(source.parameters(), target.parameters()):
        tgt_param.data.copy_(tgt_param.data * (1.0 - tau) + src_param.data * tau)


Activation = Union[str, nn.Module]

_str_to_activation = {
    'relu': nn.ReLU(),
    'tanh': nn.Tanh(),
    'identity': nn.Identity(),
    'softmax': nn.Softmax(dim=-1),
}


def mlp(sizes,
        activation: Activation = 'relu',
        output_activation: Activation = 'identity'):
    if isinstance(activation, str):
        activation = _str_to_activation[activation]
    if isinstance(output_activation, str):
        output_activation = _str_to_activation[output_activation]

    layers = []
    for i in range(len(sizes) - 1):
        act = activation if i < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[i], sizes[i + 1]), act]
    return nn.Sequential(*layers)

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


def get_reward(state, info, snake_index, reward, snake_my_delta, snake_your_delta, height, width, final_result):
    state = np.squeeze(np.array(state), axis=2)
    step_reward = np.zeros(len(snake_index))
    for i in snake_index:
        if final_result == 1:       # done and won
            step_reward[i] += 200
        elif final_result == 2:     # done and lose
            step_reward[i] -= 100
        elif final_result == 3:     # done and draw
            step_reward[i] += 20
        else:                       # not done
            if reward[i] > 0:           # eat a bean
                step_reward[i] += 30
            else:                       # just move
                snakes_position = np.array(info['snakes_position'], dtype=object)
                beans_position = np.array(info['beans_position'], dtype=object)
                snakes_len = len(snakes_position)
                snake_heads = [snake[0] for snake in snakes_position]
                self_head = np.array(snake_heads[i])
                #dists = [min(abs(self_head[0] - other_head[0]), abs(self_head[0] + other_head[0] + 2 - height)) + 
                #        min(abs(self_head[1] - other_head[1]), abs(self_head[1] + other_head[1] + 2 - width))
                #        for other_head in beans_position]
                # [np.sqrt(np.sum(np.square(other_head - self_head))) for other_head in beans_position]
                min_x, min_y, index = get_min_bean(self_head[1], self_head[0], beans_position, width, height, snakes_position, state)
                step_reward[i] -= (min_x + min_y) * 2
                step_reward[i] += snake_my_delta * 10
                if (snake_your_delta < 0): step_reward[i] += snake_your_delta * (-10)
                # step_reward[i] += min(dists)
                # step_reward[i] += (snake_my_delta - snake_your_delta) * 10
                # step_reward[i] += snake_your_delta * (-10)
                # if reward[i] < 0:
                #    step_reward[i] -= 10
    return step_reward


def logits_random(act_dim, logits):
    logits = torch.Tensor(logits).to(device)
    acs = [Categorical(out).sample().item() for out in logits]
    num_agents = len(logits)
    actions = np.random.randint(act_dim, size=num_agents << 1)
    actions[:num_agents] = acs[:]
    return actions

def append_random(act_dim, action):
    action = torch.Tensor([action]).to(device)
    acs = [out for out in action]
    num_agents = len(action)
    actions = np.random.randint(act_dim, size=num_agents << 1)
    actions[:num_agents] = acs[:]
    return actions

def logits_greedy(state, info, logits, height, width):
    state = np.squeeze(np.array(state), axis=2)
    beans = info['beans_position']
    snakes = info['snakes_position']

    logits = torch.Tensor(logits).to(device)
    logits_action = np.array([Categorical(out).sample().item() for out in logits])
    greedy_action = greedy_snake(state, beans, snakes, width, height, [1])

    action_list = np.zeros(2)
    action_list[0] = logits_action[0]
    action_list[1] = greedy_action[0]

    return action_list

def append_greedy(act_dim, state, info, action, height, width, step):
    state = np.squeeze(np.array(state), axis=2)
    beans = info['beans_position']
    snakes = info['snakes_position']

    action = torch.Tensor([action]).to(device)
    logits_action = np.array([out for out in action])
    greedy_action = greedy_snake(state, beans, snakes, width, height, [1], step)
    
    action_list = np.zeros(2)
    action_list[0] = logits_action[0]
    action_list[1] = greedy_action[0]

    return action_list

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


def save_config(args, save_path):
    file = open(os.path.join(str(save_path), 'config.yaml'), mode='w', encoding='utf-8')
    yaml.dump(vars(args), file)
    file.close()


def load_config(args, log_path):
    file = open(os.path.join(str(log_path), 'config.yaml'), "r")
    config_dict = yaml.load(file, Loader=yaml.FullLoader)
    print("@", config_dict)
    args = SN(**config_dict)
    print("@@", args)
    return args


# def set_algos():
#     with open(os.path.join(os.path.dirname(__file__), "config", "default.yaml"), "r") as f:
#         try:
#             config_dict = yaml.load(f, Loader=yaml.FullLoader)
#         except yaml.YAMLError as exc:
#             assert False, "default.yaml error: {}".format(exc)
#
#     args = SN(**config_dict)
#     return args


