import copy
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


def get_surrounding(state, width, height, x, y):
    surrounding = [state[(y - 1) % height][x],  # up
                   state[(y + 1) % height][x],  # down
                   state[y][(x - 1) % width],  # left
                   state[y][(x + 1) % width]]  # right

    return surrounding

def keep_safe(X, Y, turn, state, width, height, snakes):
    vis=np.zeros((height,width))
    from queue import Queue
    pq = Queue()
    pq.put((X,Y))
    dx = [-1,1,0,0]
    dy = [0,0,-1,1]
    mx = snakes[turn][-1][0]
    my = snakes[turn][-1][1]
    cnt = 0
    if (state[X][Y]==1): 
        mx = -1
        my = -1
    while (not pq.empty()):
        (x,y) =pq.get()
        if (vis[x][y]==1): continue
        vis[x][y] = 1
        cnt += 1
        for i in range(4):
            x1=x+dx[i]
            y1=y+dy[i]
            x1 += height
            x1 %= height
            y1 += width
            y1 %= width
            
            if (state[x1][y1]==2 or state[x1][y1]==3 and (x1 != mx or y1 != my)): continue
            pq.put((x1,y1))
    return cnt
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
    if (state[x][y]==1): mp2[x][y]=turn + 2
    elif (not (Lx==x and Ly==y)): 
        mp2[Lx][Ly]=0
        mp2[x][y]=turn + 2 
    return mp2

def Cnt_d(state, beans, snakes, width, height, turn, dir):
    mp=get_map(state,beans,snakes,width,height,turn,dir)
    dx = [-1,1,0,0]
    dy = [0,0,-1,1]
    d=np.zeros((height,width))
    for i in range(height):
        for j in range(width):
            for k in range(4):
                x = i+ dx[k]
                y = j+ dy[k]
                x += height
                y += width
                x %= height
                y %= height
                if (mp[x][y]==0 or mp[x][y]==1):
                    d[i][j] += 1
    cnt = 0 
    for i in range(height):
        for j in range(width):
            if (d[i][j]<=1 and mp[i][j]<2):
                cnt += 1
    return cnt

def Check_Circle(snakes, id, width, height):
    x = snakes[id][0][0]
    y = snakes[id][0][1]
    Lx = snakes[id][-1][0]
    Ly = snakes[id][-1][1]
    dx = [-1,1,0,0]
    dy = [0,0,-1,1]
    for i in range(4):
        x1 = x + dx[i]
        y1 = y + dy[i]
        x1 += height
        y1 += width
        x1 %= height
        y1 %= width
        if (x1==Lx and y1 == Ly): return (True,i)
    return (False,-1)

def bfs(state, target, snakes, width, height, turn):
    from queue import Queue
    # print(target,"target")
    Q = Queue()
    Q.put((-1,snakes[turn][0][0],snakes[turn][0][1],-1))  # step, x,y, fir_dir
    dx = [-1,1,0,0]
    dy = [0,0,-1,1]
    while (not Q.empty()):
        (step, x,y , dir) = Q.get()
        # print(step,x,y,dir,"(step,x,y,dir)")
        if (step > 7): return (-1,-1)
        for i in range(4):
            x1 = x + dx[i]
            y1 = y + dy[i]
            x1 += height
            y1 += width
            x1 %= height
            y1 %= width
            # print(x1,y1,"x1,y1")
            if (x1==target[step+1][0] and y1 == target[step+1][1]):
                # print(x1,y1)
                if (step==-1): return (i, step+1)
                else: return (dir, step+1)
            if (state[x1][y1]>1): continue
            if (dir == -1):
                Q.put((step+1, x1,y1,i))
            else:
                Q.put((step+1,x1,y1,dir))
    return (-1,-1)
def Get_NEW_MAP(state, Lx, Ly):
    mp2= copy.deepcopy(state)
    mp2[Lx][Ly]=0
    return mp2

def f(delta_len, my_len, d_bean, d_rear):
    if (delta_len<=0 or my_len<=11): return d_bean
    if (delta_len==1 or my_len<=12): return d_bean*0.9+d_rear*0.1
    if (delta_len==2 or my_len<=13): return d_bean*0.7+d_rear*0.3
    if (delta_len<=4 or my_len<=15): return d_bean*0.5+d_rear*0.5
    if (delta_len<=6 or my_len<=17): return d_bean*0.3+d_rear*0.7
    if (delta_len<=9 or my_len<=19): return d_bean*0.1+d_rear*0.9
    return d_rear

def greedy_snake(state_map, beans, snakes, width, height, ctrl_agent_index, Current_Step):
    beans_position = copy.deepcopy(beans)
    actions = []
    for i in ctrl_agent_index:
        head_x = snakes[i][0][0]
        head_y = snakes[i][0][1]
        Lx = snakes[i][-1][0]
        Ly = snakes[i][-1][1]
        len_my= len(snakes[i])
        len_U = len(snakes[i^1])
        (Flag, dirt) = Check_Circle(snakes,i,width,height) 
        if (len_my > len_U and len_my > 9  and Flag):
            actions.append(dirt)
            # print(777)
            return actions

        if (len_my < len_U and len_U >= 10):
            (Flag, dirt) = Check_Circle(snakes,i^1,width,height) 
            if (Flag):
                (dir,NEED_step) = bfs(state_map, snakes[i^1][-1:-10:-1],snakes,width,height,i)
                # print(666)
                # print(dir,NEED_step)
                # print()
                if (dir != -1 and NEED_step < 50-Current_Step): 
                    actions.append(dir)
                    return actions
        Map_without_my_rear = Get_NEW_MAP(state_map, Lx, Ly)
        head_surrounding = get_surrounding(Map_without_my_rear, width, height, head_y, head_x)
        bean_y, bean_x, index = get_min_bean(head_y, head_x, beans_position, width, height, snakes, state_map)
        mat= diji(state_map,bean_x,bean_y,width,height)
        mat_rear = diji(state_map, Lx, Ly,width, height)
        dx = [-1,1,0,0]
        dy = [0,0,-1,1]
        dis= np.zeros(4)
        Blok=np.zeros(4)
        D = np.zeros(4)
        Tup = []
        for i in range(4):
            head_x_tmp = head_x + dx[i]
            head_y_tmp = head_y + dy[i]
            head_x_tmp += height
            head_x_tmp %= height
            head_y_tmp += width
            head_y_tmp %= width
            if (head_surrounding[i]>1): dis[i]=1000000
            else: 
                dis[i]= f(len_my-len_U,len_my,mat[head_x_tmp][head_y_tmp], mat_rear[head_x_tmp][head_y_tmp])
            if (head_surrounding[i]>1): Blok[i]=-100000
            else: Blok[i]= keep_safe(head_x_tmp, head_y_tmp,ctrl_agent_index[0],state_map,width,height,snakes)
            if (Blok[i]<4): Blok[i]=-100000
            else: Blok[i]=55
            
            if (head_surrounding[i]>1): D[i]=-10000
            else: D[i]= Cnt_d(state_map,beans,snakes,width,height,ctrl_agent_index[0],i)
            Tup.append((Blok[i],-dis[i],D[i],i))
        actions.append(Tup.index(max(Tup))) 
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


current_step2=0
# from functools import wraps
# current_step=0
# def wrapper(func):  # func = my_controller
#     @wraps(func)
#     def inner(*args, **kwargs):
#                                          #step1
#         ret = func(*args, **kwargs)
#         global current_step
#         current_step+=1
#         print(current_step)                           #step3
#         return ret
#     return inner

# @wrapper   # my_controller = wrapper(my_controller)
def my_controller(observation_list, action_space_list, is_act_continuous=False):
    global current_step2
    current_step2 += 1
    print (current_step2,"dd")
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
    actions = greedy_snake(state, beans, snakes, width, height, [mysnake], current_step2)
    joint_action = to_joint_action(actions, 1)          #step2
    return joint_action                                  #step4
