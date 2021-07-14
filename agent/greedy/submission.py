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
    mat = diji(state,y,x,width, height)
    for i, (bean_y, bean_x) in enumerate(beans_position):
        # distance = math.sqrt((x - bean_x) ** 2 + (y - bean_y) ** 2)
        distance = mat[bean_y][bean_x]
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
                y = i+  dy[k]
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
def greedy_snake(state_map, beans, snakes, width, height, ctrl_agent_index):
    beans_position = copy.deepcopy(beans)
    actions = []
    for i in ctrl_agent_index:
        head_x = snakes[i][0][1]
        head_y = snakes[i][0][0]
        head_surrounding = get_surrounding(state_map, width, height, head_x, head_y)
        bean_x, bean_y, index = get_min_bean(head_x, head_y, beans_position, width, height, snakes, state_map)
        beans_position.pop(index)
        # print(len(snakes[i]))
        t = max (4, len(snakes[i])//2, len(snakes[i^1])//2)
        next_distances = []
        mat= diji(state_map,bean_y,bean_x,width,height)
        # mat = floyd(height, width, snakes)
        # bean_id = get_id(bean_y, bean_x, width)
        head_y_tmp = (head_y - 1) % height
        head_id_tmp = get_id(head_y_tmp, head_x, width)
        up_distance = math.inf if head_surrounding[0] > 1 or keep_safe(head_y_tmp,head_x,ctrl_agent_index[0],state_map,width,height,snakes)<t  else \
            mat[head_y_tmp][head_x]
        up_d = math.inf if head_surrounding[0] > 1 or keep_safe(head_y_tmp,head_x,ctrl_agent_index[0],state_map,width,height,snakes)<t else \
            Cnt_d(state_map,beans, snakes,width, height,ctrl_agent_index[0],0)    
            # mat[head_id_tmp][bean_id]
            # math.sqrt((head_x - bean_x) ** 2 + ((head_y - 1) % height - bean_y) ** 2)
        next_distances.append((up_distance,up_d))
        head_y_tmp = (head_y + 1) % height
        head_id_tmp = get_id(head_y_tmp, head_x, width)
        down_distance = math.inf if head_surrounding[1] > 1 or keep_safe(head_y_tmp,head_x,ctrl_agent_index[0],state_map,width,height,snakes)<t else \
            mat[head_y_tmp][head_x]
        down_d = math.inf if head_surrounding[1] > 1 or keep_safe(head_y_tmp,head_x,ctrl_agent_index[0],state_map,width,height,snakes)<t else \
            Cnt_d(state_map,beans, snakes,width, height,ctrl_agent_index[0],1)
            # math.sqrt((head_x - bean_x) ** 2 + ((head_y + 1) % height - bean_y) ** 2)
        next_distances.append((down_distance,down_d))
        head_x_tmp = (head_x - 1) % width
        head_id_tmp = get_id(head_y, head_x_tmp, width)
        left_distance = math.inf if head_surrounding[2] > 1 or keep_safe(head_y,head_x_tmp,ctrl_agent_index[0],state_map,width,height,snakes)<t else \
            mat[head_y][head_x_tmp]
            # math.sqrt(((head_x - 1) % width - bean_x) ** 2 + (head_y - bean_y) ** 2)
        left_d = math.inf if head_surrounding[2] > 1 or keep_safe(head_y,head_x_tmp,ctrl_agent_index[0],state_map,width,height,snakes)<t else \
            Cnt_d(state_map,beans, snakes,width, height,ctrl_agent_index[0],2)
        next_distances.append((left_distance,left_d))
        head_x_tmp = (head_x + 1) % width
        head_id_tmp = get_id(head_y, head_x_tmp, width)
        right_distance = math.inf if head_surrounding[3] > 1 or keep_safe(head_y,head_x_tmp,ctrl_agent_index[0],state_map,width,height,snakes)<t else \
            mat[head_y][head_x_tmp]
        right_d = math.inf if head_surrounding[3] > 1 or keep_safe(head_y,head_x_tmp,ctrl_agent_index[0],state_map,width,height,snakes)<t else \
            Cnt_d(state_map,beans, snakes,width, height,ctrl_agent_index[0],3)
            # math.sqrt(((head_x + 1) % width - bean_x) ** 2 + (head_y - bean_y) ** 2)
        next_distances.append((right_distance,right_d))
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
    actions = greedy_snake(state, beans, snakes, width, height, [mysnake])
    joint_action = to_joint_action(actions, 1)
    return joint_action
