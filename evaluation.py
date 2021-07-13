import numpy as np

from agent.dqn.rl_agent import get_observations
from agent.greedy.greedy_agent import greedy_snake
from agent.dqn.rl_agent import agent as dqn_snake
import agent.search
from env.chooseenv import make
from tabulate import tabulate
import argparse
import random
import copy

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

# def F_calc(observation_list):
#     beans_position = observation_list[1]
#     P1 = get_min_bean_distance(observation_list[3][0][1], observation_list[3][0][0], beans_position)- get_min_bean_distance(observation_list[2][0][1],observation_list[2][0][0],beans_position)
#     P2 = shape(observation_list[2])[0]-shape(observation_list[3])[0]
#     P3 = get_sum_bean_distance(observation_list[3][0][1], observation_list[3][0][0], beans_position) - get_sum_bean_distance(observation_list[2][0][1],observation_list[2][0][0], beans_position)
#     A=1
#     B=1
#     C=0.5
#     return A*P1+B*P2+C*P3

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
    return mp2

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
    observation_list['last_direction'][turn]=d1^1
    return observation_list

def reborn_map(observation_list_new,turn,mp):
    for i in range(observation_list_new['broad_height']):
        for j in range(observation_list_new['broad_width']):
            if (mp[i][j]==turn): mp[i][j]=0
    for i in range(3):
        mp[observation_list_new[turn][i][0]][observation_list_new[turn][i][1]]=turn
    return mp

# def it_dfs(d,turn,observation_list,mp):
#     if (d[turn]==0): return F_calc (observation_list)
#     d[turn]-=1
#     cnt=0
#     sum = 0
#     for i in range(4):
#         if (Check_available(observation_list,turn,i,mp)==False): continue
#         cnt += 1
#         mp_new=get_map(observation_list,turn,i,mp)
#         sum+=(it_dfs(d,turn^1, get_observation_list(observation_list,turn,i,mp,mp_new),mp_new ))
#     if (cnt==0):
#         return it_dfs(d,turn^1,reborn_list(observation_list,turn,mp),reborn_map(reborn_list(observation_list,turn,mp),turn,mp))
#     else:
#         return sum/cnt

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
'''
def get_my_action(observation_list, mp,my_snake):
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
    actions = get_my_action(observation_list[0], mp,mysnake)
    player = []
    each = [0] * 4
    each[actions[0]] = 1
    player.append(each)
    joint_action.append(player)
    return joint_action
'''
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

def get_my_action(state,beans,snakes,width,height,mysnake):
    joint_action=[]
    actions = get_my_action2(state, beans, snakes, width, height, mysnake)
    return actions

def print_state(state, actions, step):
    state = np.array(state)
    state = np.squeeze(state, axis=2)
    print(f'----------------- STEP:{step} -----------------')
    print(f'state:\n{state}')
    print(f'actions: {actions}\n')


def get_actions(obs, algo, greedy_info, side):
 #   print(obs, type(obs))
    actions = np.random.randint(4, size=1)

    # dqn
 #   print(greedy_info, type(greedy_info))
    if algo == 'my':
        
        temp = get_my_action(greedy_info['state'],
                                  greedy_info['beans'],
                                  greedy_info['snakes'],
                                  greedy_info['width'],
                                  greedy_info['height'], side)
   #     print(actions, temp)
        actions[:] = temp[:]

    elif algo == 'greedy':
        if side == 0:
            ctrl_agent_index = [0]
        else:
            ctrl_agent_index = [1]

        actions[:] = greedy_snake(greedy_info['state'],
                                  greedy_info['beans'],
                                  greedy_info['snakes'],
                                  greedy_info['width'],
                                  greedy_info['height'], ctrl_agent_index)[:]
 
    return actions


def join_actions(obs, algo_list, greedy_info):
    first_action = get_actions(obs[0], algo_list[0], greedy_info, side=0)
    second_action = get_actions(obs[1], algo_list[1], greedy_info, side=1)
    actions = np.zeros(2)
    actions[0] = first_action[:]
    actions[1] = second_action[:]
    return actions


def run_game(env, algo_list, episode, verbose=False):
    width = env.board_width
    height = env.board_height
    obs_dim = 18
    agent_index = [0, 1]
    total_reward = np.zeros(2)
    num_win = np.zeros(3)
    
    for i in range(1, episode + 1):
        episode_reward = np.zeros(2)
        state, info = env.reset()
 #       print(np.asarray(state).reshape(-1, 1).shape, 'state, info')
        obs = get_observations(state, info, agent_index, obs_dim, height, width)

        greedy_info = {'state': np.squeeze(np.array(state), axis=2), 'beans': info['beans_position'],
                       'snakes': info['snakes_position'], 'width': width, 'height': height}
        action_list = join_actions(obs, algo_list, greedy_info)
        joint_action = env.encode(action_list)

        step = 0
        if verbose:
            print_state(state, action_list, step)

        while True:
            next_state, reward, done, _, info = env.step(joint_action)
            episode_reward += reward

            if done:
                if np.sum(episode_reward[0]) > np.sum(episode_reward[1]):
                    num_win[0] += 1
                elif np.sum(episode_reward[0]) < np.sum(episode_reward[1]):
                    num_win[1] += 1
                else:
                    num_win[2] += 1

                if not verbose:
                    print('.', end='')
                    if i % 100 == 0 or i == episode:
                        print()
                break

            state = next_state
            step += 1
            obs = get_observations(state, info, agent_index, obs_dim, height, width)

            greedy_info = {'state': np.squeeze(np.array(state), axis=2), 'beans': info['beans_position'],
                           'snakes': info['snakes_position'], 'width': width, 'height': height}

            action_list = join_actions(obs, algo_list, greedy_info)
            joint_action = env.encode(action_list)

            if verbose:
                print_state(state, action_list, step)

        total_reward += episode_reward

    # calculate results
    total_reward /= episode
    print(f'\nResult base on {episode} ', end='')
    print('episode:') if episode == 1 else print('episodes:')

    header = ['Name', algo_list[0], algo_list[1]]
    data = [['score', total_reward[0], total_reward[1]],
            ['win', num_win[0], num_win[1]]]
    print(tabulate(data, headers=header, tablefmt='pretty', floatfmt='.3f'))


if __name__ == "__main__":
    env_type = 'snakes_1v1'

    game = make(env_type, conf=None)

    parser = argparse.ArgumentParser()
    parser.add_argument("--my_ai", default="my", help="dqn/random/greedy")
    parser.add_argument("--opponent", default="greedy", help="dqn/random/greedy")
    parser.add_argument("--episode", default=100)
    args = parser.parse_args()

    # [greedy, dqn, random]
    agent_list = [args.my_ai, args.opponent]
    run_game(game, algo_list=agent_list, episode=args.episode, verbose=False)
