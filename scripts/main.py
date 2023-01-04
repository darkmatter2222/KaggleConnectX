from kaggle_environments import evaluate, make, utils
import pandas as pd
import numpy as np
import random
import requests
from random import choice
from tqdm import tqdm
from scipy.signal import convolve2d
import sys
from io import StringIO

import os, sys
from pathlib import Path


import keras
model2 = keras.models.load_model('/kaggle_simulations/agent/test5')

#result = requests.get('https://www.kaggleusercontent.com/episodes/45908179.json')
#sys.stdout.write('googleresult')
#sys.stdout.write(str(len(result.text)))

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
    sys.stdout.write(f'desiered_target: {desiered_target}')
    board_copy = board.copy()
    row = np.argmax(np.argwhere(board_copy[:,desiered_target] == 0))
    board_copy[row, desiered_target] = me
    
    if row > 0:
        board_copy[row - 1, desiered_target] = enemy
        enemy_win = winning_move(board_copy, enemy)
        if enemy_win:
            targets = np.argwhere(board_copy[:1] == 0)[:,1].tolist()
            try:
                targets.remove(desiered_target)
            except:
                pass
                
            if len(targets) > 0:
                target = choice(targets)
                sys.stdout.write('avoideable...')
            else:
                sys.stdout.write('gaurenteed predicted loss or stalemate')
                target = desiered_target
        else:
            target = desiered_target
            sys.stdout.write('safe to proceed...')
    else:
        target = desiered_target
        sys.stdout.write('safe to proceed... at 0')
    return target

def smart_move(board, me, enemy, recurse_count = 0, target_chosen=0):
    sys.stdout.write(f'me:{me} enemy:{enemy}')
    
    target_chosen = target_chosen
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
            perform_rand_move = False
            flat_board = board.flatten().tolist()
            hold = flat_board
            predicted = False
            try:
                hold = tuple(flat_board)
                flat_board.append(me)
                
                result = np.argmax(model2.predict([flat_board]))
                row = np.argwhere(board[:,result] == 0)
                if len(row) > 0:
                    me_row_land = np.argmax(row) # Lowest possible placement possible
                    result = permit_move_filter(me, enemy, board, result)
                    board[me_row_land, result] = me
                    target_chosen = int(result)
                    sys.stdout.write(f'smart move {result}')
                else:
                    perform_rand_move = True
                    sys.stdout.write(f'invalid move {result}')
            except:
                sys.stdout.write(f'FAIL')
                perform_rand_move = True
            
            if perform_rand_move:
                me_col_choice = random.choice(targets) # random placement verticaly based on top row
                me_row_land = np.argmax(np.argwhere(board[:,me_col_choice] == 0)) # Lowest possible placement possible
                me_col_choice = permit_move_filter(me, enemy, board, me_col_choice)
                board[me_row_land, me_col_choice] = me
                target_chosen = me_col_choice
                sys.stdout.write(f'rand move {me_col_choice}')
    return board, target_chosen


# This agent random chooses a non-empty column.
def my_agent(observation, configuration):
    from random import choice
    board = np.resize(observation.board,(6,7))
    #return choice([c for c in range(configuration.columns) if observation.board[c] == 0])
    me = 2
    enemy = 1   
    if observation.mark == 1:
        sys.stdout.write('i went first')
        me = 1
        enemy = 2
    elif observation.mark == 2:
        sys.stdout.write('enemy first')
        me = 2
        enemy = 1    
    
    board, target = smart_move(board, me, enemy)
    return target
