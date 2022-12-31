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
    
    board, target = smart_move(board, me, enemy)

    return target
