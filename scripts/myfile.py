# coding: utf-8
from kaggle_environments import evaluate, make
import numpy as np
from scipy.signal import convolve2d
import random, json, gym, sys
from os.path import exists
class ConnectX(gym.Env):
    def __init__(self, switch_prob=0.5):
        self.env = make('connectx', debug=True)
        self.pair = [None, 'negamax']
        self.trainer = self.env.train(self.pair)
        self.switch_prob = switch_prob
        
        # Define required gym fields (examples):
        config = self.env.configuration
        self.action_space = gym.spaces.Discrete(config.columns)
        self.observation_space = gym.spaces.Discrete(config.columns * config.rows)

    def switch_trainer(self):
        self.pair = self.pair[::-1]
        self.trainer = self.env.train(self.pair)

    def step(self, action):
        return self.trainer.step(action)
    
    def reset(self):
        if random.uniform(0, 1) < self.switch_prob:
            self.switch_trainer()
        return self.trainer.reset()
    
    def render(self, **kwargs):
        return self.env.render(**kwargs)

class QTable:
    def __init__(self, action_space):
        self.file = 'qtable.json'
        self.load_table()
        self.action_space = action_space
        
    def add_item(self, state_key):
        self.table[state_key] = list(np.zeros(self.action_space.n))
        
    def __call__(self, state):
        board = state['board'][:] # Get a copy
        board.append(state.mark)
        state_key = np.array(board).astype(str)
        state_key = hex(int(''.join(state_key), 3))[2:]
        if state_key not in self.table.keys():
            self.add_item(state_key)
        
        return self.table[state_key]
    
    def load_table(self, ):
        if exists(self.file):
            f = open(self.file)
            self.table = json.loads(f.read())
            f.close()
        else:
            self.table = dict()

    def save_qtable(self,):
        f = open(self.file,"w")
        document = json.dumps(self.table)
        f.write(document)
        f.close()
env = ConnectX()
q_table = QTable(env.action_space)
horizontal_kernel = np.array([[ 1, 1, 1, 1]])
vertical_kernel = np.transpose(horizontal_kernel)
diag1_kernel = np.eye(4, dtype=np.uint8)
diag2_kernel = np.fliplr(diag1_kernel)
detection_kernels = [horizontal_kernel, vertical_kernel, diag1_kernel, diag2_kernel]

def winning_move(board, player):
    for kernel in detection_kernels:
        if (convolve2d(board == player, kernel, mode="valid") == 4).any():
            return True
    return False

def permit_move_filter(me, enemy, board, desiered_target):
    target = 0
    board_copy = board.copy()
    row = np.argmax(np.argwhere(board_copy[:,desiered_target] == 0))
    board_copy[row, desiered_target] = me
    
    enemy_win = winning_move(board_copy, enemy)
    if enemy_win:
        targets = np.argwhere(board_copy[:1] == 0)[:,1].tolist()
        targets.remove(desiered_target)
        if len(targets) > 0:
            target = choice(targets)
        else:
            target = desiered_target
    else:
        target = desiered_target
    return target
    

def my_agent(observation, configuration):
    from random import choice
    board = np.resize(observation.board,(6,7))
    #return choice([c for c in range(configuration.columns) if observation.board[c] == 0])
    me = 2
    enemy = 1   
    if observation.mark == 1:
        me = 1
        enemy = 2
    elif observation.mark == 2:
        me = 2
        enemy = 1    
    
    sys.stdout.write(f'me:{me} enemy:{enemy}')
    
    target_chosen = 0
    targets = np.argwhere(board[:1] == 0)[:,1].tolist()
    if len(targets) == 0:
        raise ValueError("stale")
    if len(targets) > 1:
        random.shuffle(targets)
    sim_win = False
    sim_win_target = 0
    for target in targets:
        board_copy = board.copy()
        me_row_land = np.argmax(np.argwhere(board[:,target] == 0)) # Lowest possible placement possible
        board_copy[me_row_land, target] = me
        sim_me_win = winning_move(board_copy, me)
        if sim_me_win:
            sim_win = True
            sim_win_target = target
            break
    if sim_win:
        me_row_land = np.argmax(np.argwhere(board[:,sim_win_target] == 0)) # Lowest possible placement possible
        board[me_row_land, sim_win_target] = me
        target_chosen = sim_win_target
        sys.stdout.write(f'attack! {sim_win_target}')
    else:
        sim_loss = False
        sim_loss_target = 0
        
        for target in targets:
            board_copy = board.copy()
            enemy_row_land = np.argmax(np.argwhere(board[:,target] == 0)) # Lowest possible placement possible
            board_copy[enemy_row_land, target] = enemy
            sim_me_loss = winning_move(board_copy, enemy)
            if sim_me_loss:
                sim_loss = True
                sim_loss_target = target
                break
        if sim_loss:
            me_row_land = np.argmax(np.argwhere(board[:,sim_loss_target] == 0)) # Lowest possible placement possible
            board[me_row_land, sim_loss_target] = me
            target_chosen = sim_loss_target
            sys.stdout.write(f'Defend! {sim_loss_target}')
        else:
            perform_rand_move = True
            flat_board = board.flatten().tolist()
            hold = flat_board
            predicted = False
            
            board_v2 = observation.board[:]
            board_v2.append(observation.mark)
            state_key = list(map(str, board_v2))
            state_key = hex(int(''.join(state_key), 3))[2:]
            result = 0
            if state_key in q_table.table.keys():
                 result = int(np.argmax(q_table.table[state_key]))
            row = np.argwhere(board[:,result] == 0)
            if len(row) > 0:
                me_row_land = np.argmax(row) # Lowest possible placement possible
                result = permit_move_filter(me, enemy, board, result)
                board[me_row_land, result] = me
                
                target_chosen = int(result)
                sys.stdout.write(f'smart move {result}')
                perform_rand_move = False
            else:
                sys.stdout.write(f'invalid move from Qtable {result}')
            
            if perform_rand_move:
                me_col_choice = random.choice(targets) # random placement verticaly based on top row
                print(me_col_choice)
                me_row_land = np.argmax(np.argwhere(board[:,me_col_choice] == 0)) # Lowest possible placement possible
                me_col_choice = permit_move_filter(me, enemy, board, me_col_choice)
                board[me_row_land, me_col_choice] = me
                target_chosen = me_col_choice
                sys.stdout.write(f'rand move {me_col_choice}')
    return target_chosen
