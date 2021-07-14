
import random
from types import CodeType

from numpy.core.fromnumeric import shape
from numpy.core.numeric import zeros_like
import math
import numpy as np

def get_id(x, y, width):
    return x * width + y
    

def floyd(height, width, snakes):
    mat = np.zeros((height * width,height * width))
    for x1 in range(height):
        for y1 in range(width):
            for x2 in range(height):
                for y2 in range(width):
                    id1 = get_id(x1, y1, width)
                    id2 = get_id(x2, y2, width)
                    if (id1 != id2):
                        mat[id1][id2] = math.inf
                        mat[id2][id1] = math.inf
                    else : mat[id1][id2] = 0
    for x in range(height):
        for y in range(width):
            id1 = get_id(x, y, width)
            if (x == 0) : id2 = id1 + (height - 1) * width
            else : id2 = id1 - width
            mat[id1][id2] = 1
            if (y == 0) : id2 = id1 + width - 1
            else : id2 = id1 - 1
            mat[id1][id2] = 1
            if (x == height - 1) : id2 = id1 - (height - 1) * width
            else : id2 = id1 + width
            mat[id1][id2] = 1
            if (y == width - 1) : id2 = id1 - width + 1
            else : id2 = id1 + 1
            mat[id1][id2] = 1
    for i, (snake_x, snake_y) in enumerate(snakes[0]):
        id1 = get_id(snake_x,snake_y,width)
        for x in range(height):
            for y in range(width):
                id2 = get_id(x, y ,width)
                if (id1 != id2):
                    mat[id1][id2] = math.inf
                    mat[id2][id1] = math.inf
    for i, (snake_x, snake_y) in enumerate(snakes[1]):
        id1 = get_id(snake_x,snake_y,width)
        for x in range(height):
            for y in range(width):
                id2 = get_id(x, y ,width)
                if (id1 != id2):
                    mat[id1][id2] = math.inf
                    mat[id2][id1] = math.inf
    sum_id = height * width
    for k in range(sum_id):
        for i in range(sum_id):
            for j in range(sum_id):
                if (mat[i][k] + mat[k][j] < mat[i][j]):
                    mat[i][j] = mat[i][k] + mat[k][j]
    return mat
 
def get_min_bean_distance(x, y, beans_position, width, height, snakes):
    min_distance = math.inf
    if (shape(beans_position)[0]==0): return 0
    min_x = beans_position[0][0]
    min_y = beans_position[0][1]
    index = 0
    # mat = floyd(height, width, snakes)
    for i, (bean_x, bean_y) in enumerate(beans_position):
        distance = min(abs(x - bean_x), abs(x + bean_x + 2 - height))  + min(abs (y - bean_y), abs(y + bean_y + 2 - width))
        # snake_id = get_id(x, y, width)
        # beans_id = get_id(bean_x, bean_y, width)
        # distance = mat[snake_id][beans_id]
        if distance < min_distance:
            min_x = bean_x
            min_y = bean_y
            min_distance = distance
            index = i
    return min_distance

def get_sum_bean_distance(x, y, beans_position, width, height, snakes):
    distance = 0
    # mat = floyd(height, width, snakes)
    for i, (bean_x, bean_y) in enumerate(beans_position):
        distance += min(abs(x - bean_x), abs(x + bean_x + 2 - height))  + min(abs (y - bean_y), abs(y + bean_y + 2 - width))
        # snake_id = get_id(x, y, width)
        # beans_id = get_id(bean_x, bean_y, width)
        # tmp = mat[snake_id][beans_id]
        # if (tmp != math.inf) : distance += tmp
    return distance

def F_calc(state, bean, snakes, width, height):
    P1 = get_min_bean_distance(snakes[1][0][0],snakes[1][0][1],bean, width, height, snakes)-get_min_bean_distance(snakes[0][0][0],snakes[0][0][1],bean,width,height, snakes)
    P2 = shape(snakes[0])[0]-shape(snakes[1])[0]
    P3 = get_sum_bean_distance(snakes[1][0][0],snakes[1][0][1],bean, width, height, snakes)-get_sum_bean_distance(snakes[0][0][0],snakes[0][0][1],bean, width, height, snakes)
    A= 2
    B =1
    C = 0.5
    return A*P1 + B*P2 + C*P3

def get_min_bean_distance_index(x, y, beans_position, width, height, snakes):
    min_distance = math.inf
    if (shape(beans_position)[0]==0): return 0
    min_x = beans_position[0][0]
    min_y = beans_position[0][1]
    index = 0
    for i, (bean_x, bean_y) in enumerate(beans_position):
        distance = min(abs(x - bean_x), abs(x + bean_x + 2 - height))  + min(abs (y - bean_y), abs(y + bean_y + 2 - width))
        # snake_id = get_id(x, y, width)
        # beans_id = get_id(bean_x, bean_y, width)
        # distance = floyd(height, width, snakes)[snake_id][beans_id]
        if distance < min_distance:
            min_x = bean_x
            min_y = bean_y
            min_distance = distance
            index = i
    return  index


def F_calc_greedy_hacker(state, bean, snakes, width, height):
    your_dist= get_min_bean_distance(snakes[1][0][0], snakes[1][0][1], bean, width, height, snakes)
    your_bean = get_min_bean_distance_index(snakes[1][0][0], snakes[1][0][1], bean, width, height, snakes)
    my_dist = get_min_bean_distance(snakes[0][0][0],snakes[0][0][1],bean, width, height, snakes)
    my_bean =  get_min_bean_distance_index(snakes[0][0][0],snakes[0][0][1],bean, width, height, snakes)
    if (my_bean == your_bean and my_dist < your_dist) : P4 = 1
    else : P4 = 0
    P1 = your_dist - my_dist
    P2 = shape(snakes[0])[0]-shape(snakes[1])[0]
    P3 = get_sum_bean_distance(snakes[1][0][0], snakes[1][0][1], bean, width, height, snakes) - get_sum_bean_distance(snakes[0][0][0],snakes[0][0][1], bean, width, height, snakes)
    A=4
    B=6
    C=1
    D=0
    return A*P1+B*P2+C*P3+D*P4

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

'''def it_dfs(d,turn,state,bean, snakes, width, height):
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
        return sum/cnt'''
def diji(state, X, Y, width, height):
    mp=np.zeros((width,height))
    for i in range(height):
        for j in range(width):
            mp[i][j]=math.inf
    mp[X][Y]=0
    vis=np.zeros((width,height))
    from queue import PriorityQueue as PQ
    pq=PQ()
    pq.push((0,(X,Y)))
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
                pq.push((mp[x1][y1],(x1,y1)))
    return mp


def it_dfs(d,turn,state,bean, snakes, width, height):
    if (d[turn]==0 or shape(bean)[0]<=3): return F_calc_greedy_hacker(state, bean, snakes, width, height)
    d[turn] -= 1
    cnt = 0
    sum = 0
    ls = []
    for i in range(4):
        if (Check_available(state, bean, snakes, width, height,turn,i)):
            cnt+=1
            ls.append(it_dfs(d,turn^1,get_map(state,bean,snakes,width,height,turn,i),get_beans(state,bean,snakes,width,height,turn,i),get_snakes(state,bean,snakes,width,height,turn,i),width,height))
    if (cnt==0):
        snakes = reborn_snake(state,bean,snakes,width,height,turn,i)
        return it_dfs(d,turn^1,reborn_state(state,bean,snakes,width,height,turn,i),bean,snakes,width,height)
    else:
        ls.sort(reverse = True)
        if (cnt==1): return ls[0]
        elif (cnt==2):
            if (turn ==0):
                return 0.75*ls[0]+0.25*ls[1]
            else: return 0.25*ls[0]+0.75*ls[1]
        else:
            if (turn==0):
                return 0.7*ls[0]+0.2*ls[1]+0.1*ls[2]
            else: return 0.1*ls[0]+0.2*ls[1]+0.7*ls[2]
def it_dfs_min_max(d,turn,state,bean, snakes, width, height,MinMax):
    if (d[turn]==0 or shape(bean)[0]<=3): return F_calc_greedy_hacker(state, bean, snakes, width, height)
    d[turn] -= 1
    cnt = 0
    This_MIN_MAX=1
    if (turn==0): This_MIN_MAX=-100000
    else: This_MIN_MAX = 100000
    ls = []
    for i in range(4):
        if (Check_available(state, bean, snakes, width, height,turn,i)):
            cnt += 1
            tmp= it_dfs_min_max(d,turn^1,get_map(state,bean,snakes,width,height,turn,i),get_beans(state,bean,snakes,width,height,turn,i),get_snakes(state,bean,snakes,width,height,turn,i),width,height,This_MIN_MAX)
            if (turn==0):
                if (tmp>This_MIN_MAX): This_MIN_MAX=tmp
                if (This_MIN_MAX>=MinMax): return This_MIN_MAX
            else:
                if (tmp<This_MIN_MAX): This_MIN_MAX=tmp
                if (This_MIN_MAX<=MinMax): return This_MIN_MAX
    if (cnt==0):
        snakes = reborn_snake(state,bean,snakes,width,height,turn,i)
        return it_dfs_min_max(d,turn^1,reborn_state(state,bean,snakes,width,height,turn,i),bean,snakes,width,height,This_MIN_MAX)
    else:
        return This_MIN_MAX

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

def get_my_action_MINMAX(state,beans,snakes,width,height,my_snake):
    if (my_snake==0):
        ans=-111111
        dir=0
        for i in range(4):
            if (Check_available(state,beans,snakes,width,height,my_snake,i)):
                mp_new=get_map(state,beans, snakes, width, height, my_snake,i)
                tmp = it_dfs_min_max([12,12],my_snake^1,mp_new,get_beans(state,beans,snakes,width,height,my_snake,i),snakes,width,height,-100000)
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
                tmp = it_dfs_min_max([12,12],my_snake^1,mp_new,get_beans(state,beans,snakes,width,height,my_snake,i),snakes,width,height,100000)
                if (tmp<ans):
                    ans=tmp
                    dir=i
        return [dir]

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
                tmp = it_dfs([8,8],my_snake^1,mp_new,get_beans(state,beans,snakes,width,height,my_snake,i),snakes,width,height)
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
    actions = get_my_action_MINMAX(state, beans, snakes, width, height, mysnake)
    player = []
    each = [0] * 4
    each[actions[0]] = 1
    player.append(each)
    joint_action.append(player)
    return joint_action
