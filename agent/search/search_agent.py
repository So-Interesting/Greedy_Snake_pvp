from random import random
import copy

from numpy.core.fromnumeric import shape
from numpy.core.numeric import zeros_like
import math
import numpy as np


    
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
    dx = [-1,1,0,0]
    dy = [0,0,-1,1]
    x += dx[dir]
    y += dy[dir]
    x += observation_list['broad_height']
    x %= observation_list['broad_height']
    y += observation_list['broad_width']
    y %= observation_list['broad_width']
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
    dx = [-1,1,0,0]
    dy = [0,0,-1,1]
    x += dx[dir]
    y += dy[dir]
    x += observation_list['broad_height']
    x %= observation_list['broad_height']
    y += observation_list['broad_width']
    y %= observation_list['broad_width']

    Lx = observation_list[turn][-1][0]
    Ly = observation_list[turn][-1][1]
    for i in range(observation_list['broad_height']):
        for j in range(observation_list['broad_width']):
            if (mp_new[i][j]==1): bean_list.append([i,j])
    if (mp[x][y]!=1): observation_list[turn].pop()
    observation_list['last_direction'][turn]=dir

def reborn_list(observation_list,turn,mp):
    dx=[-1,1,0,0]
    dy=[0,0,-1,1]

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
    for i in range(4):
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
    mp=np.zeros((observation_list['board_height'],observation_list['board_width']))
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

def get_action(observation_list, mp,my_snake):
    dx = [-1,1,0,0]
    dy = [0,0,-1,1]
    if (my_snake==2):
        ans=-111
        dir=0
        for i in range(4):
            if (Check_available(observation_list,my_snake,i,mp)):
                mp_new=get_map(observation_list,2,i,mp)
                tmp = it_dfs([6,6],3,get_observation_list(observation_list,2,i,mp,mp_new),mp_new)
                if (tmp>ans):
                    ans=tmp
                    dir=i
        return dir
    else:
        ans=1111111
        dir=0
        for i in range(4):
            if (Check_available(observation_list,my_snake,i,mp)):
                mp_new=get_map(observation_list,3,i,mp)
                tmp = it_dfs([6,6],2,get_observation_list(observation_list,3,i,mp,mp_new),mp_new)
                if (tmp<ans):
                    ans=tmp
                    dir=i
        return dir
def my_controller(observation_list, action_space_list, is_act_continuous=False):
    joint_action = []
    mysnake = observation_list[0]['controlled_snake_index']
    mp=Map_All(observation_list[0])
    actions = get_action(observation_list[0], mp,mysnake)
    player = []
    each = [0] * 4
    each[actions[0]] = 1
    player.append(each)
    joint_action.append(player)
    return joint_action
