import copy
import math
import numpy as np
import pandas as pd


    
def get_min_bean_distance(x, y, beans_position):
    min_distance = math.inf
    min_x = beans_position[0][0]
    min_y = beans_position[0][1]
    index = 0
    for i, (bean_y, bean_x) in enumerate(beans_position):
        distance = math.abs(x - bean_x)  + math.abs (y - bean_y)
        if distance < min_distance:
            min_x = bean_x
            min_y = bean_y
            min_distance = distance
            index = i
    return min_distance

def get_sum_bean_distance(x, y, beans_position):
    distance = 0
    for i, (bean_y, bean_x) in enumerate(beans_position):
        distance += math.abs(x - bean_x)  + math.abs (y - bean_y)
    return distance

def F_calc(observation_list, snake, eat):
    beans_position = observation_list[1]
    P1 = get_min_bean_distance(snake[3].x, snake[3].y, beans_position)- get_min_bean_distance(snake[2].x,snake[2].y,beans_position)
    P2 = eat[2] - eat[3]
    P3 = get_sum_bean_distance(snake[3].x, snake[3].y, beans_position) - get_sum_bean_distance(snake[2].x,snake[2].y, beans_position)
    A=1
    B=1
    C=0.5
    return A*P1+B*P2+C*P3

class SNAKE:
    def __init__ (self,L,d):
        self.x = L[0][0] # head of a snake
        self.y = L[0][1]
        self.L = L # the whole body
        self.d = d # the direct of the snake


# def it_dfs(d, turn, state_map,Min_Max):
#     if (d[turn]==0): return dirc.default, F_calc
#     d[turn]-=1
#     MINMAX = inf/-inf
#     Choose a turn
#         dir,X = it_dfs(d,turn^1,state_map2,MINMAX)
#         if (MINMAX Min_Max) => return
#         choose a min/max answer
#     return dirc, Min/Max F_calc

def Map_All(observation_list):
    mp=np.zeros(observation_list['board_height'],observation_list['board_width'])
    for i in observation_list[1]:
        mp[i[0],i[1]]=1
    t=0
    for i in observation_list[2]:
        if (t==0): 
            mp[i[0],i[1]]=-2
            t=1
        else: mp[i[0],i[1]]=2
    t=0
    for i in observation_list[3]:
        if (t==0):    
            mp[i[0],i[1]]=-3
            t=1
        else: mp[i[0],i[1]]=2
    return mp

def sample(observation_list, action_space_list_each, is_act_continuous):
    state_map= Map_All(observation_list)
    s = np.zeros(3,dtype=SNAKE)
    s[2] = SNAKE(observation_list[2],observation_list['last_direction'][2])
    s[3] = SNAKE(observation_list[3],observation_list['last_direction'][3])
    player = []
    for j in range(len(action_space_list_each)):
        # each = [0] * action_space_list_each[j]
        # idx = np.random.randint(action_space_list_each[j])
        if action_space_list_each[j].__class__.__name__ == "Discrete":
            each = [0] * action_space_list_each[j].n
            idx = action_space_list_each[j].sample()
            each[idx] = 1
            player.append(each)
        elif action_space_list_each[j].__class__.__name__ == "MultiDiscreteParticle":
            each = []
            nvec = action_space_list_each[j].high
            sample_indexes = action_space_list_each[j].sample()

            for i in range(len(nvec)):
                dim = nvec[i] + 1
                new_action = [0] * dim
                index = sample_indexes[i]
                new_action[index] = 1
                each.extend(new_action)
            player.append(each)
    return player

def my_controller(observation_list, action_space_list, is_act_continuous=False):
    joint_action = []
    for i in range(len(action_space_list)):
        player = sample(observation_list, action_space_list[i], is_act_continuous)
        joint_action.append(player)
    return joint_action


def get_min_bean(x, y, beans_position):
    min_distance = math.inf
    min_x = beans_position[0][1]
    min_y = beans_position[0][0]
    index = 0
    for i, (bean_y, bean_x) in enumerate(beans_position):
        distance = math.sqrt((x - bean_x) ** 2 + (y - bean_y) ** 2)
        if distance < min_distance:
            min_x = bean_x
            min_y = bean_y
            min_distance = distance
            index = i
    return min_x, min_y, index


def get_surrounding(state, width, height, x, y):
    surrounding = [state[(y - 1) % height][x],  # up
                   state[(y + 1) % height][x],  # down
                   state[y][(x - 1) % width],  # left
                   state[y][(x + 1) % width]]  # right

    return surrounding


def greedy_snake(state_map, beans, snakes, width, height, ctrl_agent_index):
    beans_position = copy.deepcopy(beans)
    actions = []
    for i in ctrl_agent_index:
        head_x = snakes[i][0][1]
        head_y = snakes[i][0][0]
        head_surrounding = get_surrounding(state_map, width, height, head_x, head_y)
        bean_x, bean_y, index = get_min_bean(head_x, head_y, beans_position)
        beans_position.pop(index)

        next_distances = []
        up_distance = math.inf if head_surrounding[0] > 1 else \
            math.sqrt((head_x - bean_x) ** 2 + ((head_y - 1) % height - bean_y) ** 2)
        next_distances.append(up_distance)
        down_distance = math.inf if head_surrounding[1] > 1 else \
            math.sqrt((head_x - bean_x) ** 2 + ((head_y + 1) % height - bean_y) ** 2)
        next_distances.append(down_distance)
        left_distance = math.inf if head_surrounding[2] > 1 else \
            math.sqrt(((head_x - 1) % width - bean_x) ** 2 + (head_y - bean_y) ** 2)
        next_distances.append(left_distance)
        right_distance = math.inf if head_surrounding[3] > 1 else \
            math.sqrt(((head_x + 1) % width - bean_x) ** 2 + (head_y - bean_y) ** 2)
        next_distances.append(right_distance)
        actions.append(next_distances.index(min(next_distances)))
    return actions


def to_joint_action(actions, num_agent):
    joint_action = []
    for i in range(num_agent):
        action = actions[i]
        one_hot_action = [0] * 4
        one_hot_action[action] = 1
        one_hot_action = [one_hot_action]
        joint_action.append(one_hot_action)
    return joint_action
