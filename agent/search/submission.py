
import random

from numpy.core.fromnumeric import shape
from numpy.core.numeric import zeros_like
import math
import numpy as np


def get_min_bean_distance(x, y, beans_position):
    min_distance = math.inf
    if (shape(beans_position)[0]==0): return 0
    min_x = beans_position[0][0]
    min_y = beans_position[0][1]
    index = 0
    for i, (bean_x, bean_y) in enumerate(beans_position):
        distance = abs(x - bean_x)  + abs (y - bean_y)
        if distance < min_distance:
            min_x = bean_x
            min_y = bean_y
            min_distance = distance
            index = i
    return min_distance

def get_sum_bean_distance(x, y, beans_position):
    distance = 0
    for i, (bean_x, bean_y) in enumerate(beans_position):
        distance += abs(x - bean_x)  + abs (y - bean_y)
    return distance

def F_calc(state, bean, snakes, width, height):
    P1 = get_min_bean_distance(snakes[1][0][0],snakes[1][0][1],bean)-get_min_bean_distance(snakes[0][0][0],snakes[0][0][1],bean)
    P2 = shape(snakes[0])[0]-shape(snakes[1])[0]
    P3 = get_sum_bean_distance(snakes[1][0][0],snakes[1][0][1],bean)-get_sum_bean_distance(snakes[0][0][0],snakes[0][0][1],bean)
    A= 2
    B =1
    C = 0.5
    return A*P1 + B*P2 + C*P3

def get_min_bean_distance_index(x, y, beans_position):
    min_distance = math.inf
    if (shape(beans_position)[0]==0): return 0
    min_x = beans_position[0][0]
    min_y = beans_position[0][1]
    index = 0
    for i, (bean_x, bean_y) in enumerate(beans_position):
        distance = abs(x - bean_x)  +  abs (y - bean_y)
        if distance < min_distance:
            min_x = bean_x
            min_y = bean_y
            min_distance = distance
            index = i
    return  index


def F_calc_greedy_hacker(state, bean, snakes, width, height):
    your_dist= get_min_bean_distance(snakes[1][0][0], snakes[1][0][1], bean)
    your_bean = get_min_bean_distance_index(snakes[1][0][0], snakes[1][0][1], bean)
    my_dist = get_min_bean_distance(snakes[0][0][0],snakes[0][0][1],bean)
    my_bean =  get_min_bean_distance_index(snakes[0][0][0],snakes[0][0][1],bean)
    if (my_bean == your_bean and my_dist < your_dist) : P4 = 1
    else : P4 = 0
    P1 = your_dist - my_dist
    P2 = shape(snakes[0])[0]-shape(snakes[1])[0]
    P3 = get_sum_bean_distance(snakes[1][0][0], snakes[1][0][1], bean) - get_sum_bean_distance(snakes[0][0][0],snakes[0][0][1], bean)
    A=1
    B=1
    C=0.5
    D=4
    return A*P1+B*P2+C*P3+D*P4


def Check_available(observation_list,turn,dir,mp):
    x = observation_list[turn][0][1]
    y = observation_list[turn][0][0]
    dx = [0,0,-1,1]
    dy = [-1,1,0,0]
    x+=dx[dir]
    y+=dy[dir]
    x += observation_list['broad_weight']
    x %= observation_list['broad_weight']
    y+=observation_list['broad_height']
    y%=observation_list['broad_height']
    
    Lx = observation_list[turn][-1][1]
    Ly = observation_list[turn][-1][0]
    if (mp[y][x]==0 or mp[y][x]==1 or (x==Lx and y==Ly)): return True
    else: return False

def get_beans(state,bean,snakes,width,height,turn, dir):
    x = snakes[turn][0][0]
    y = snakes[turn][0][1]
    dx = [-1,1,0,0]
    dy = [0,0,-1,1]
    x += dx[dir]
    y += dy[dir]
    x += height
    x %= height
    y += width
    y %= width
    beans=list()
    for [i,j] in bean:
        if (not (i==x and j==y)):
            beans.append([i,j])
    return beans

def get_snakes(state,bean,snakes,width,height,turn, dir):
    x = snakes[turn][0][0]
    y = snakes[turn][0][1]
    dx = [-1,1,0,0]
    dy = [0,0,-1,1]
    x += dx[dir]
    y += dy[dir]
    x += height
    x %= height
    y += width
    y %= width
    snakes[turn].insert(0,[x,y])
    if (state[x][y]!=1): snakes[turn].pop()
    return snakes

def reborn_snake(state,bean, snakes, width,height, turn, i):
    dx=[-1,1,0,0]
    dy=[0,0,-1,1]

    x = random.randrange(0, height)
    y = random.randrange(0, width)
    d1=random.randrange(0,4)    
    x1=x+dx[d1]
    y1=y+dy[d1]
    d2=random.randrange(0,4)
    x2=x1+dx[d2]
    y2=y1+dy[d2]
    x1+=height
    x1%=height
    x2+=height
    x2%=height
    y1+=width
    y1%=width
    y2+=width
    y2%=width
    while ((state[x][y]!=0 and state[x][y]!=turn) or (state[x1][y1]!=0 and state[x1][y1]!=turn) or (state[x2][y2]!=0 and state[x2][y2]!=turn) or (x==x2 and y==y2)):
        x = random.randrange(0, height)
        y = random.randrange(0, width)
        d1=random.randrange(0,4)    
        x1=x+dx[d1]
        y1=y+dy[d1]
        d2=random.randrange(0,4)
        x2=x1+dx[d2]
        y2=y1+dy[d2]
        x1+=height
        x1%=height
        x2+=height
        x2%=height
        y1+=width
        y1%=width
        y2+=width
        y2%=width
    snakes[turn]=[[x,y],[x1,y1],[x2,y2]]
    return snakes

def reborn_state(state,bean,snakes,width,height,turn,dir):
    mp=np.zeros((height,width))
    for i in bean:
        mp[i[0],i[1]]=1
    for i in snakes[0]:
        mp[i[0],i[1]]=2
    for i in snakes[1]:
        mp[i[0],i[1]]=3
    return mp

def it_dfs(d,turn,state,bean, snakes, width, height):
    if (d[turn]==0): return F_calc_greedy_hacker(state, bean, snakes, width, height)
    d[turn] -= 1
    cnt = 0
    sum = 0
    for i in range(4):
        if (Check_available(state, bean, snakes, width, height,turn,i)):
            cnt+=1
            sum+=it_dfs(d,turn^1,get_map(state,bean,snakes,width,height,turn,i),get_beans(state,bean,snakes,width,height,turn,i),get_snakes(state,bean,snakes,width,height,turn,i),width,height)
    if (cnt==0):
        snakes = reborn_snake(state,bean,snakes,width,height,turn,i)
        return it_dfs(d,turn^1,reborn_state(state,bean,snakes,width,height,turn,i),bean,snakes,width,height)
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

def get_map(state, beans, snakes, width, height, turn, dir):
    mp2=state.copy()
    x= snakes[turn][0][0]
    y = snakes[turn][0][1]
    dx = [-1,1,0,0]
    dy = [0,0,-1,1]
    x += dx[dir]
    y += dy[dir]
    x += height
    x %= height
    y += width
    y %= width
    Lx = snakes[turn][-1][0]
    Ly = snakes[turn][-1][1]
    if (state[x][y]==1): mp2[x][y]=turn
    elif (not (Lx==x and Ly==y)): 
        mp2[Lx][Ly]=0
        mp2[x][y]=turn
    return mp2

def Check_available(states,beans,snakes,width,height,turn,dir):
    x = snakes[turn][0][1]
    y = snakes[turn][0][0]
    dx = [0,0,-1,1]
    dy = [-1,1,0,0]
    x+=dx[dir]
    y+=dy[dir]
    x += width
    x %= width
    y+= height
    y%= height
    
    Lx = snakes[turn][-1][1]
    Ly = snakes[turn][-1][0]
    if (states[y][x]==0 or states[y][x]==1 or (x==Lx and y==Ly)): return True
    else: return False

def get_my_action2(state, beans ,snakes, width, height, my_snake):
    dx = [-1,1,0,0]
    dy = [0,0,-1,1]
    if (my_snake==0):
        ans=-111
        dir=0
        for i in range(4):
            if (Check_available(state,beans,snakes,width,height,my_snake,i)):
                mp_new=get_map(state,beans, snakes, width, height, my_snake,i)
                tmp = it_dfs([8,8],my_snake^1,get_map(state,beans,snakes,width,height,my_snake,i),get_beans(state,beans,snakes,width,height,my_snake,i),snakes,width,height)
                if (tmp>ans):
                    ans=tmp
                    dir=i
        return [dir]
    else:
        ans=1111111
        dir=0
        for i in range(4):
            if (Check_available(state,beans,snakes,width,height,my_snake,i)):
                mp_new=get_map(state,beans,snakes,width, height,my_snake,i)
                tmp = it_dfs([8,8],my_snake^1,)
                if (tmp<ans):
                    ans=tmp
                    dir=i
        return [dir]

def my_controller(observation_list, action_space_list, is_act_continuous=False):
    joint_action = []
    width = observation_list[0]['board_width']
    height = observation_list[0]['board_height']
    mysnake = observation_list[0]['controlled_snake_index']
    state = np.zeros((height, width))
    beans = observation_list[0][1]
    snakes = []
    snakes.append(observation_list[0][2])
    snakes.append(observation_list[0][3])
    for i in beans:
        state[i[0], i[1]] = 1
    for i in snakes[0]:
        state[i[0], i[1]] = 2
    for i in snakes[1]:
        state[i[0], i[1]] = 3
    actions = get_my_action2(state, beans, snakes, width, height, mysnake)
    player = []
    each = [0] * 4
    each[actions[0]] = 1
    player.append(each)
    joint_action.append(player)
    return joint_action
