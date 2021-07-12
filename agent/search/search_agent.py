from random import random
from agent.dqn.rl_agent import get_observations
import copy

from numpy.core.fromnumeric import shape
from numpy.core.numeric import zeros_like
from env.snakes import Snake
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

def F_calc(observation_list):
    beans_position = observation_list[1]
    P1 = get_min_bean_distance(observation_list[3][0][0], observation_list[3][0][1], beans_position)- get_min_bean_distance(observation_list[2][0][0],observation_list[2][0][1],beans_position)
    P2 = shape(observation_list[2])[0]-shape(observation_list[3])[0]
    P3 = get_sum_bean_distance(observation_list[3][0][0], observation_list[3][0][1], beans_position) - get_sum_bean_distance(observation_list[2][0][0],observation_list[2][0][1], beans_position)
    A=1
    B=1
    C=0.5
    return A*P1+B*P2+C*P3

def Check_available(observation_list,turn,dir,mp):
    x = observation_list[turn][0][0]
    y = observation_list[turn][0][1]
    if (dir==-2):
        x -= 1
        x += observation_list['broad_height']
        x %= observation_list['broad_height']
    elif (dir==2):
        x+=1
        x%= observation_list['broad_height']
    elif (dir==-1):
        y-=1
        y+=observation_list['broad_width']
        y%=observation_list['broad_width']
    else:
        y+=1
        y%=observation_list['broad_width']
    Lx = observation_list[turn][-1][0]
    Ly = observation_list[turn][-1][1]
    if (mp[x][y]==0 or mp[x][y]==1 or (x==Lx and y==Ly)): return True
    else: return False

def get_map(observation_list,turn,i,mp):
    mp2=mp.copy()
    x= observation_list[turn][0][0]
    y = observation_list[turn][0][1]
    if (dir==-2):
        x -= 1
        x += observation_list['broad_height']
        x %= observation_list['broad_height']
    elif (dir==2):
        x+=1
        x%= observation_list['broad_height']
    elif (dir==-1):
        y-=1
        y+=observation_list['broad_width']
        y%=observation_list['broad_width']
    else:
        y+=1
        y%=observation_list['broad_width']
    Lx = observation_list[turn][-1][0]
    Ly = observation_list[turn][-1][1]
    if (mp[x][y]==1): mp2[x][y]=turn
    elif (not (Lx==x and Ly==y)): 
        mp2[Lx][Ly]=0
        mp2[x][y]=turn

def get_observation_list(observation_list, turn, dir, mp, mp_new):
    bean_list=list()
    x= observation_list[turn][0][0]
    y = observation_list[turn][0][1]
    if (dir==-2):
        x -= 1
        x += observation_list['broad_height']
        x %= observation_list['broad_height']
    elif (dir==2):
        x+=1
        x%= observation_list['broad_height']
    elif (dir==-1):
        y-=1
        y+=observation_list['broad_width']
        y%=observation_list['broad_width']
    else:
        y+=1
        y%=observation_list['broad_width']
    Lx = observation_list[turn][-1][0]
    Ly = observation_list[turn][-1][1]
    for i in range(observation_list['broad_height']):
        for j in range(observation_list['broad_width']):
            if (mp_new[i][j]==1): bean_list.append([i,j])
    if (mp[x][y]!=1): observation_list[turn].pop()
    observation_list['last_direction'][turn]=dir

def reborn_list(observation_list,turn,mp):
    dx=[0,0,1,-1]
    dy=[1,-1,0,0]

    x = random.randrange(0, observation_list['board_height'])
    y = random.randrange(0, observation_list['board_width'])
    d1=random.randrange(0,4)    
    x1=x+dx[d1]
    y1=y+dy[d1]
    d2=random.randrange(0,4)
    x2=x1+dx[d2]
    y2=y1+dy[d2]
    x1+=observation_list['board_height']
    x1%=observation_list['board_height']
    x2+=observation_list['board_height']
    x2%=observation_list['board_height']
    y1+=observation_list['board_width']
    y1%=observation_list['board_width']
    y2+=observation_list['board_width']
    y2%=observation_list['board_width']
    while ((mp[x][y]!=0 and mp[x][y]!=turn) or (mp[x1][y1]!=0 and mp[x1][y1]!=turn) or (mp[x2][y2]!=0 and mp[x2][y2]!=turn) or (x==x2 and y==y2)):
        x = random.randrange(0, observation_list['board_height'])
        y = random.randrange(0, observation_list['board_width'])
        d1=random.randrange(0,4)        
        x1=x+dx[d1]
        y1=y+dy[d1]
        d2=random.randrange(0,4)
        x2=x1+dx[d2]
        y2=y1+dy[d2]
        x1+=observation_list['board_height']
        x1%=observation_list['board_height']
        x2+=observation_list['board_height']
        x2%=observation_list['board_height']
        y1+=observation_list['board_width']
        y1%=observation_list['board_width']
        y2+=observation_list['board_width']
        y2%=observation_list['board_width']
    observation_list[turn]=[[x,y],[x1,y1],[x2,y2]]
    observation_list['last_direction'][turn]=2*(x-x1)+(y-y1)
    return observation_list

def reborn_map(observation_list_new,turn,mp):
    for i in range(observation_list_new['broad_height']):
        for j in range(observation_list_new['broad_width']):
            if (mp[i][j]==turn): mp[i][j]=0
    for i in range(3):
        mp[observation_list_new[turn][i][0]][observation_list_new[turn][i][1]]=turn
    return mp

def it_dfs(d,turn,observation_list,mp):
    if (d[turn]==0): return F_calc (observation_list)
    d[turn]-=1
    cnt=0
    sum = 0
    for i in range(-2,3):
        if (i==0): continue
        if (Check_available(observation_list,turn,i,mp)==False): continue
        cnt += 1
        mp_new=get_map(observation_list,turn,i,mp)
        sum+=(it_dfs(d,turn^1, get_observation_list(observation_list,turn,i,mp,mp_new),mp_new ))
    if (cnt==0):
        return it_dfs(d,turn^1,reborn_list(observation_list,turn,mp),reborn_map(reborn_list(observation_list,turn,mp),turn,mp))
    else:
        return sum/cnt
        
    

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
        player = sample(observation_list[0], action_space_list[i], is_act_continuous)
        joint_action.append(player)
    return joint_action


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
