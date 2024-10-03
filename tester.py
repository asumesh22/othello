from main import *
import random
import time

class best_move:
    def __init__(self):
        self.bestmove = -1

    def value(self, n):
        self.bestmove = n
        return n

def play_game():
    board = '.'*27 + 'ox......xo' + '.'*27
    smart_player = random.choice(['x', 'o'])
    random_player = ENEMY[smart_player]
    tkn = 'x'

    while True:
        if tkn == smart_player: # get a smart move
            strategy = Strategy()
            bm = strategy.best_strategy(board, smart_player, best_move(), 0, 0)
            if bm == -1: # swap players
                tkn = ENEMY[tkn]
                rm = strategy.random_strategy(board, random_player, best_move(), 0, 0)
                if rm == -1: break
                else: continue
            player1, player2 = string_to_bitboards(board, tkn)
            player1, player2 = make_move(2**(63-bm), player1, player2)
            board = bitboards_to_string(player1, player2, tkn)
            tkn = ENEMY[tkn]
        elif tkn == random_player: # get a random move
            strategy = Strategy()
            rm = strategy.random_strategy(board, random_player, best_move(), 0, 0)
            if rm == -1: # swap players
                tkn = ENEMY[tkn]
                bm = strategy.best_strategy(board, smart_player, best_move(), 0, 0)
                if bm == -1: break
                else: continue
            player1, player2 = string_to_bitboards(board, tkn)
            player1, player2 = make_move(2**(63-rm), player1, player2)
            board = bitboards_to_string(player1, player2, tkn)
            tkn = ENEMY[tkn]
                
    return board, smart_player, random_player   

def main():
    t1 = time.time()
    NUM_GAMES = 100
    me, you = 0, 0
    for game_num in range(1, NUM_GAMES+1):
        board, smart, rand = play_game()
        print(f'{game_num} - diff {board.count(smart)-board.count(rand)}; score {board.count(smart)} vs {board.count(rand)} as {smart}')
        me += board.count(smart)
        you += board.count(rand)
    print(100*(me/(me+you)), f'in {time.time()-t1}s')

if __name__ == '__main__': main()