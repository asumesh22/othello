# import
import math
import random

# globals
global EAST_MASK; EAST_MASK = 0b11111110_11111110_11111110_11111110_11111110_11111110_11111110_11111110
global WEST_MASK; WEST_MASK = 0b01111111_01111111_01111111_01111111_01111111_01111111_01111111_01111111
global ENEMY; ENEMY = {'x':'o', 'o':'x'}
global WEIGHTS; WEIGHTS = [15, 5, 5, 1, 1, 10, 5]
global CORNERS; CORNERS = {0, 7, 56, 63}
global X_POS; X_POS = {9, 14, 49, 54}
global C_POS; C_POS = {1, 6, 8, 15, 48, 55, 57, 62}

# caches
global MOVESCACHE; MOVESCACHE = {}
global ALPHABETACACHE; ALPHABETACACHE = {}
global TERMINALCACHE; TERMINALCACHE = {}
global STABLEDISKSCACHE; STABLEDISKSCACHE = {}
global MAKEMOVESCACHE; MAKEMOVESCACHE = {}
global TOSTRINGCACHE; TOSTRINGCACHE = {}

# trivial functions
def display_board(bitboard1, bitboard2, tkn): # Displays the board in two dimensions.
    board = bitboards_to_string(bitboard1, bitboard2, tkn)

    for row_num in range(8):
        print(' '.join([*board[row_num*8:row_num*8+8]]))

def bitboards_to_string(bitboard1, bitboard2, tkn): # Turns two bitboards into a string.
    key = (bitboard1, bitboard2, tkn)
    if key in TOSTRINGCACHE: return TOSTRINGCACHE[key]

    board = ''

    for i in range(64):
        if at(bitboard1, i): # black tkn at i -> add tkn
            board += tkn
        elif at(bitboard2, i): # white tkn at i -> add enemy tkn
            board += ENEMY[tkn]
        else: # empty square at i -> add dot
            board += '.'

    TOSTRINGCACHE[key] = board
    return board

def string_to_bitboards(board, tkn): # Reverse of bitboards_to_string.
    player1 = 0
    player2 = 0

    for i, val in enumerate(board):
        if val == tkn: player1 += 2**(63-i)
        elif val == ENEMY[tkn]: player2 += 2**(63-i)
        
    return (player1, player2)

def bit_num_to_int(bitNum): # Turns a binary number into an int.
    return int(math.log2(bitNum))

# basic functionality
def at(bitboard, idx): # Returns the token in bitboard at idx.
    return (bitboard >> (63-idx)) & 1

def shiftN(bb): return bb << 8
def shiftE(bb): return (bb & EAST_MASK) >> 1
def shiftS(bb): return bb >> 8
def shiftW(bb): return (bb & WEST_MASK) << 1
def shiftNW(bb): return shiftN(shiftW(bb))
def shiftNE(bb): return shiftN(shiftE(bb))
def shiftSW(bb): return shiftS(shiftW(bb))
def shiftSE(bb): return shiftS(shiftE(bb))

def get_moves(bitboard1, bitboard2): # Returns a list of the possible moves given two bitboards.
    if key := (bitboard1, bitboard2) in MOVESCACHE:
        return MOVESCACHE[key]
    
    moves = 0b0
    empty = ~(bitboard1 | bitboard2)

    # north
    captured = shiftN(bitboard1) & bitboard2
    for i in range(5): captured |= shiftN(captured) & bitboard2
    moves |= shiftN(captured) & empty
    
    # south
    captured = shiftS(bitboard1) & bitboard2
    for i in range(5): captured |= shiftS(captured) & bitboard2
    moves |= shiftS(captured) & empty

    # west
    captured = shiftW(bitboard1) & bitboard2
    for i in range(5): captured |= shiftW(captured) & bitboard2
    moves |= shiftW(captured) & empty

    # east 
    captured = shiftE(bitboard1) & bitboard2
    for i in range(5): captured |= shiftE(captured) & bitboard2
    moves |= shiftE(captured) & empty

    # northwest
    captured = shiftNW(bitboard1) & bitboard2
    for i in range(5): captured |= shiftNW(captured) & bitboard2
    moves |= shiftNW(captured) & empty

    # northeast
    captured = shiftNE(bitboard1) & bitboard2
    for i in range(5): captured |= shiftNE(captured) & bitboard2
    moves |= shiftNE(captured) & empty

    # southwest
    captured = shiftSW(bitboard1) & bitboard2
    for i in range(5): captured |= shiftSW(captured) & bitboard2
    moves |= shiftSW(captured) & empty

    # southeast
    captured = shiftSE(bitboard1) & bitboard2
    for i in range(5): captured |= shiftSE(captured) & bitboard2
    moves |= shiftSE(captured) & empty

    moves_list = []
    
    for i in range(64):
        if ((moves>>i) & 1):
            moves_list.append(2**(63-bit_num_to_int((1<<i))))

    return moves_list

def make_move(move, bitboard1, bitboard2): # Makes the move and returns the boards.
    key = (move, bitboard1, bitboard2)
    if key in MAKEMOVESCACHE: return MAKEMOVESCACHE[key]

    bitboard1 |= move

    # north
    captured = shiftN(move) & bitboard2
    for i in range(5): captured |= shiftN(captured) & bitboard2
    if (shiftN(captured) & bitboard1): bitboard1 |= captured; bitboard2 &= ~captured
    
    # south
    captured = shiftS(move) & bitboard2
    for i in range(5): captured |= shiftS(captured) & bitboard2
    if (shiftS(captured) & bitboard1): bitboard1 |= captured; bitboard2 &= ~captured

    # west
    captured = shiftW(move) & bitboard2
    for i in range(5): captured |= shiftW(captured) & bitboard2
    if (shiftW(captured) & bitboard1): bitboard1 |= captured; bitboard2 &= ~captured

    # east 
    captured = shiftE(move) & bitboard2
    for i in range(5): captured |= shiftE(captured) & bitboard2
    if (shiftE(captured) & bitboard1): bitboard1 |= captured; bitboard2 &= ~captured

    # northwest
    captured = shiftNW(move) & bitboard2
    for i in range(5): captured |= shiftNW(captured) & bitboard2
    if (shiftNW(captured) & bitboard1): bitboard1 |= captured; bitboard2 &= ~captured

    # northeast
    captured = shiftNE(move) & bitboard2
    for i in range(5): captured |= shiftNE(captured) & bitboard2
    if (shiftNE(captured) & bitboard1): bitboard1 |= captured; bitboard2 &= ~captured

    # southwest
    captured = shiftSW(move) & bitboard2
    for i in range(5): captured |= shiftSW(captured) & bitboard2
    if (shiftSW(captured) & bitboard1): bitboard1 |= captured; bitboard2 &= ~captured

    # southeast
    captured = shiftSE(move) & bitboard2
    for i in range(5): captured |= shiftSE(captured) & bitboard2
    if (shiftSE(captured) & bitboard1): bitboard1 |= captured; bitboard2 &= ~captured
    
    MAKEMOVESCACHE[key] = (bitboard1, bitboard2)
    return (bitboard1, bitboard2)

# algorithms
def TERMINAL(player1, player2, alpha, beta, tkn):
    if player1 + player2 == (2**64)-1: # Board is completely filled out.
        board = bitboards_to_string(player1, player2, tkn)
        return [board.count(tkn) - board.count(ENEMY[tkn])]

    key = (player1, player2, tkn)
    if key in TERMINALCACHE:
        n, best = TERMINALCACHE[key]

        if n['lowerbound'] >= beta: return [n['lowerbound']] # + best[1:]
        if n['upperbound'] <= alpha: return [n['upperbound']] # + best[1:]

        alpha = max(alpha, n['lowerbound'])
        beta = min(beta, n['upperbound'])

        if alpha + 1 >= beta: return best

    moves = get_moves(player1, player2)
    if not moves: # player1 has no moves.
        moves = get_moves(player2, player1)

        if not moves: # player2 also has no moves (game over).
            board = bitboards_to_string(player1, player2, tkn)
            return [board.count(tkn) - board.count(ENEMY[tkn])]
        
        else: # swap players and continue
            ab = TERMINAL(player2, player1, -beta, -alpha, ENEMY[tkn])
            return [-ab[0]] + ab[1:] + [-1] # pass back up

    best = [alpha-1]
    for mv in move_ordering(moves, player1, player2, tkn):
        move = bit_num_to_int(mv)
        new_player1, new_player2 = make_move(2**(63-move), player1, player2)
        ab = TERMINAL(new_player2, new_player1, -beta, -alpha, ENEMY[tkn])
        ab_eval = -ab[0]
        if ab_eval < alpha: continue
        if ab_eval > beta: return [ab_eval]
        best = [ab_eval] + ab[1:] + [move]
        alpha = ab_eval + 1
    
    n = {'upperbound' : float('inf'), 'lowerbound' : -float('inf')}
    g = best[0]
    if g <= alpha: n['upperbound'] = g
    if g >  alpha and g < beta: n['lowerbound'] = g; n['upperbound'] = g
    if g >= beta: n['lowerbound'] = g
    TERMINALCACHE[key] = (n, best)

    return best

def HEURISTIC(player1, player2, tkn):
    board = bitboards_to_string(player1, player2, tkn)

    # corners
    corners = sum(at(player1, c) - at(player2, c) for c in CORNERS)

    # x positions
    x_positions = 0
    for pos in X_POS:
        pos_corner = -1
        if pos == 9: pos_corner = 0
        elif pos == 14: pos_corner = 7
        elif pos == 49: pos_corner = 56
        elif pos == 54: pos_corner = 63

        pos_val = [[0, 1][at(player2, pos)], -1][at(player1, pos)]
        
        if pos_val == 0: continue
        if at(player1, pos_corner): x_positions += 1
        elif at(player2, pos_corner): x_positions -= 1
        else: x_positions += pos_val

    # c positions
    c_positions = 0
    for pos in C_POS:
        pos_corner = -1
        if pos in {1, 8}: pos_corner = 0
        elif pos in {6, 15}: pos_corner = 7
        elif pos in {48, 57}: pos_corner = 56
        elif pos in {55, 62}: pos_corner = 63

        pos_val = [[0, 1][at(player2, pos)], -1][at(player1, pos)]
        
        if pos_val == 0: continue
        if at(player1, pos_corner): c_positions += 1
        elif at(player2, pos_corner): c_positions -= 1
        else: c_positions += pos_val

    # mobility
    mobility = len(get_moves(player1, player2)) - len(get_moves(player2, player1))

    # stability
    stability = stable_disks(board, tkn) - stable_disks(board, ENEMY[tkn])

    frontier = 0

    edgescore = 0
    edges = {board[0:8], board[56:], board[0::8], board[7::8]}
    for edge in edges:
        if edge == f'{tkn*8}': edgescore += 10
        elif edge == f'{ENEMY[tkn]*8}': edgescore -= 10
        elif edge == f'..{tkn*4}..': edgescore += 4
        elif edge == f'..{ENEMY[tkn]*4}..': edgescore -= 4


    # dot product
    vector = (corners, x_positions, c_positions, mobility, stability, frontier, edgescore)
    return sum(vector[i] * WEIGHTS[i] for i in range(len(vector)))

def MIDGAME(player1, player2, alpha, beta, tkn, depth):
    if depth == 0: # don't search anymore.
        return [HEURISTIC(player1, player2, tkn)]

    key = (player1, player2, tkn, depth)
    if key in ALPHABETACACHE:
        n, best = ALPHABETACACHE[key]

        if n['lowerbound'] >= beta: return [n['lowerbound']] # + best[1:]
        if n['upperbound'] <= alpha: return [n['upperbound']] # + best[1:]

        alpha = max(alpha, n['lowerbound'])
        beta = min(beta, n['upperbound'])

        if alpha + 1 >= beta: return best

    moves = get_moves(player1, player2)
    if not moves: # player1 has no moves.
        moves = get_moves(player2, player1)

        if not moves: # player2 also has no moves (game over).
            board = bitboards_to_string(player1, player2, tkn)
            return [1000 * (board.count(tkn) - board.count(ENEMY[tkn]))]
        
        else: # swap players and continue
            ab = MIDGAME(player2, player1, -beta, -alpha, ENEMY[tkn], depth-1)
            return [-ab[0]] + ab[1:] + [-1] # pass back up
       
    best = [alpha-1]
    for mv in move_ordering(moves, player1, player2, tkn):
        move = bit_num_to_int(mv)
        new_player1, new_player2 = make_move(2**(63-move), player1, player2)
        ab = MIDGAME(new_player2, new_player1, -beta, -alpha, ENEMY[tkn], depth-1)
        ab_eval = -ab[0]
        if ab_eval < alpha: continue
        if ab_eval > beta: return [ab_eval]
        best = [ab_eval] + ab[1:] + [move]
        alpha = ab_eval + 1
    
    n = {'upperbound' : float('inf'), 'lowerbound' : -float('inf')}
    g = best[0]
    if g <= alpha: n['upperbound'] = g
    if g >  alpha and g < beta: n['lowerbound'] = g; n['upperbound'] = g
    if g >= beta: n['lowerbound'] = g
    ALPHABETACACHE[key] = (n, best)

    return best

def RANDOM(player1, player2):
    moves = get_moves(player1, player2)
    random_move = random.choice(moves)
    return bit_num_to_int(random_move)

# evaluations
def stable_disks(board, tkn):
    key = (board, tkn)
    if key in STABLEDISKSCACHE: return STABLEDISKSCACHE[key]

    stableDiscs = [[0]*8 for x in range(8)]
    toProcess = [x for x in {0, 7, 56, 63} if board[x]==tkn]

    while (toProcess):
        idx = toProcess.pop()
        
        nbrs = [x for x in {idx-9, idx-8, idx-7, idx-1, idx+1, idx+7, idx+8, idx+9} if 0<=x<64]

        for nbr in nbrs:
            if stableDiscs[nbr//8][nbr%8] == 1: continue

            stable = False
            if (board[nbr] == tkn):
                stable = horizontal(nbr, stableDiscs)
                if stable:
                    stable = vertical(nbr, stableDiscs)
                if stable:
                    stable = leftDiagonal(nbr, stableDiscs)
                if stable:
                    stable = rightDiagonal(nbr, stableDiscs)
                if stable:
                    stableDiscs[nbr//8][nbr%8] = 1
                    toProcess.append(nbr)
    
    STABLEDISKSCACHE[key] = sum(x==1 for arr in stableDiscs for x in arr)
    return STABLEDISKSCACHE[key]

def horizontal(idx, stableDiscs):
    r = idx//8; c = idx%8
    if c-1<0: return True
    if stableDiscs[r][c-1] == 1: return True
    if c+1>7: return True
    if stableDiscs[r][c+1] == 1: return True
    return False

def vertical(idx, stableDiscs):
    r = idx//8; c = idx%8
    if r-1 < 0: return True
    if stableDiscs[r-1][c] == 1: return True
    if r+1>7: return True
    if stableDiscs[r+1][c] == 1: return True
    return False

def leftDiagonal(idx, stableDiscs):
    r = idx//8; c = idx % 8

    if (r-1 < 0 or c-1 < 0): return True
    if (stableDiscs[r-1][c-1] == 1): return True
    if (r+1>7 or c+1>7): return True
    if (stableDiscs[r+1][c+1] == 1): return True
    return False

def rightDiagonal(idx, stableDiscs):
    r = idx//8; c = idx%8

    if (r-1 < 0 or c+1 > 7): return True
    if (stableDiscs[r-1][c+1] == 1): return True
    if (r+1 > 7 or c<1): return True
    if (stableDiscs[r+1][c-1] == 1): return True
    return False

def iterative_deepening(board, bbSelf, bbEnemy, player, best_move):
    endgame_hl = 11
    midgame_hl = 60
    depth = 5
    max_dist = 12

    if (bc:=board.count('.')) <= endgame_hl:
        mv = TERMINAL(bbSelf, bbEnemy, -65, 65, player)
        best_move.value = mv[-1]

    elif bc <= midgame_hl:
        while depth <= max_dist:
            midgame = MIDGAME(bbSelf, bbEnemy, -float('inf'), float('inf'), player, depth)
            print(f'depth {depth} score {midgame[0]}')
            best_move.value = midgame[-1]
            depth += 1

def move_ordering(moves, bbSelf, bbEnemy, tkn):
    moves = sorted([(-ruleOfThumb(bbSelf, bbEnemy, bit_num_to_int(mv), tkn), mv) for mv in moves])
    return [c[1] for c in moves]

def ruleOfThumb(bbSelf, bbEnemy, mv, tkn):
    board = bitboards_to_string(bbSelf, bbEnemy, tkn)

    eTkn = ENEMY[tkn]
    if board.count(tkn) == 0: return -1000000
    elif board.count(eTkn) == 0: return 1000000

    weights = [
        100,-20,10, 5, 5,10,-20,100,
        -20,-50,-2,-2,-2,-2,-50,-20,
         10, -2,-1,-1,-1,-1, -2, 10,
          5, -2,-1,-1,-1,-1, -2,  5,
          5, -2,-1,-1,-1,-1, -2,  5,
         10, -2,-1,-1,-1,-1, -2, 10,
        -20,-50,-2,-2,-2,-2,-50,-20,
        100,-20,10, 5, 5,10,-20,100
    ]

    m = -10
    if board[0] == board[2] != '.': weights[1] *= m
    if board[0] == board[16] != '.': weights[8] *= m

    if board[7] == board[23] != '.': weights[15] *= m
    if board[7] == board[5] != '.': weights[6] *= m

    if board[56] == board[40] != '.': weights[48] *= m
    if board[56] == board[58] != '.': weights[57] *= m

    if board[63] == board[47] != '.': weights[55] *= m
    if board[63] == board[61] != '.': weights[62] *= m

    return weights[mv]

# main
class Strategy:
    logging = True

    def best_strategy(self, board, player, best_move, still_running, time_limit):
        player1, player2 = string_to_bitboards(board, player)

        if not get_moves(player1, player2): return -1

        iterative_deepening(board, player1, player2, player, best_move)
        
        return best_move.value

    def random_strategy(self, board, player, best_move, still_running, time_limit):
        player1, player2 = string_to_bitboards(board, player)

        if not get_moves(player1, player2): return -1
        best_move.value = RANDOM(player1, player2)
        
        return best_move.value