"""
I have two repositories.  Repo 1 contains 5 files relating to the cat trap game.  Three of them (CatGame.py, CatTrap.py, and hexutil.py)
are for the original version of the course.  Two of them (main.py and cat_trap_algorithms.py) are for the revised version of the course.
If you want to run the game, use my Repository 2 files (main.py and cat_trap_algorithms.py).  This code is consistent with the revised 
version of the course, which went live 3-11-2025.  With my 2 files (main.py and cat_trap_algorithms.py), I was able to find success. 
With my original 3 files (CatTrap.py, CatGame.py, hexutil.py), I was unable to run the game, so I spoke with Eduardo Corpeno via email.  
He advised that I may have success running the game by using this revised version, which included GitHub Codespaces.  

He mentioned that users were having trouble installing and setting up the environment, which is why he created the revised version.  
I ultimately found success getting the game to run by using GitHub Codespaces.  I was able to play the game.  In order to run the game,
one must run the main.py file and use Control + Shift + P, "Start Cat Trap Game".  You should run main.py from my Repository 2, 
which is entitled Week 1 Cat Game Code Assignment Repo 2.  Do this in Codespaces at Github.
"""

import hexutil
import random
import copy
import time
import numpy as np


def hex_to_ij(hex):
    return hex.y, hex.x//2

def ij_to_hex(i,j):
    return hexutil.Hex(2*j+(i%2),i)

class InvalidMove(ValueError):
    pass

class Game(object):
    def __init__(self, size):
        self.cat_i = size//2
        self.cat_j = size//2
        self.size  = size
        self.tiles = np.array([[0 for col in range(size)] for row in range(size)])
        self.deadline = 0
        self.terminated = False
        self.start_time = time.time()
        self.eval_fn = CatEvalFn()
        self.reached_maxdepth = False 

    def init_random_blocks(self,cat):
        n = random.randint(round(0.067*(self.size**2)),round(0.13*(self.size**2)))
        count = 0
        the_blocks=[]
        self.cat_i,self.cat_j = hex_to_ij(cat)
        self.tiles[self.cat_i][self.cat_j] = 6
        
        while count < n:
            i = random.randint(0,self.size-1)
            j = random.randint(0,self.size-1)
            if self.tiles[i][j]==0:
                self.tiles[i][j] = 1
                count = count + 1    

    def init_blocks(self,the_blocks,cat):
        i,j=hex_to_ij(cat)
        self.tiles[i][j] = 6
        
        for block in the_blocks:
            if block != [i,j]:
                self.game.tiles[block[0]][block[1]] = 1

    def CustomCat(self,randcat,ab,DLS,max_depth,ID,alotted_time):
        self.reached_maxdepth = False 
        self.start_time = time.time()
        self.deadline = self.start_time + alotted_time 
                         
        if randcat:
            result = self.RandomCat()    
        elif DLS:
            result = self.DepthLimitedCat(max_depth=max_depth, ab=ab)
        elif ID:
            self.deadline = self.start_time + alotted_time
            result = self.IterativeDeepeningCat(ab=ab)
        else:
            result = self.AlphaBetaCat() if ab else self.MinimaxCat()
            
        elapsed_time = (time.time() - self.start_time) * 1000
        print ("Elapsed time: %.3fms " % elapsed_time)
        return result

    def RandomCat(self):
        moves=self.valid_moves() #["W","E","SE","SW","NE","NW"]
        print(moves)
        dir="NONE"

        if len(moves)>0:    
            dir = random.choice(moves)
        else: 
            return [self.cat_i,self.cat_j]

        return self.target(self.cat_i,self.cat_j,dir)

    def MinimaxCat(self):
        move, placeholder = self.minimax()    
        return move 

    def AlphaBetaCat(self):
        move, placeholder = self.alphabeta()    
        return move 

    def DepthLimitedCat(self,max_depth,ab):
        move, placeholder = self.alphabeta(max_depth=max_depth) if ab else self.minimax(max_depth=max_depth)   
        return move 

    def IterativeDeepeningCat(self,ab):
        move, placeholder = self.iterative_deepening(ab)    
        return move 

       
    def valid_moves(self):
        tiles,cat_i,cat_j=self.tiles,self.cat_i,self.cat_j
        size = self.size
        moves=[]
        if (cat_j<size-1 and tiles[cat_i][cat_j+1]==0):
            moves.append("E")
        if (cat_j>0           and tiles[cat_i][cat_j-1]==0):
            moves.append("W")

        if (cat_i%2)==0:
            if (cat_i>0  and cat_j<size   and tiles[cat_i-1][cat_j]==0):
                moves.append("NE")
        else:
            if (cat_i>0  and cat_j<size-1 and tiles[cat_i-1][cat_j+1]==0):
                moves.append("NE")

        if (cat_i%2)==0:
            if (cat_i>0  and cat_j>0  and tiles[cat_i-1][cat_j-1]==0):
                moves.append("NW")
        else:
            if (cat_i>0  and cat_j>=0 and tiles[cat_i-1][cat_j]==0):
                moves.append("NW")

        if (cat_i%2)==0:
            if (cat_i<size-1  and cat_j<size   and tiles[cat_i+1][cat_j]==0):
                moves.append("SE")
        else:
            if (cat_i<size-1  and cat_j<size-1 and tiles[cat_i+1][cat_j+1]==0):
                moves.append("SE")

        if (cat_i%2)==0:
            if (cat_i<size-1  and cat_j>0  and tiles[cat_i+1][cat_j-1]==0):
                moves.append("SW")
        else:
            if (cat_i<size-1  and cat_j>0   and tiles[cat_i+1][cat_j]==0):
                moves.append("SW")
        return moves


    def target(self,i,j,dir):
        out=[i,j] 
        if   dir == "E":
            out=[i,j+1]
        elif dir =="W":
            out=[i,j-1]
        elif dir == "NE":
            out = [i-1,j] if (i%2)==0 else [i-1,j+1]
        elif dir == "NW":
            out=[i-1,j-1] if (i%2)==0 else [i-1,j]
        elif dir == "SE":
            out=[i+1,j]   if (i%2)==0 else [i+1,j+1]            
        elif dir == "SW":    
            out=[i+1,j-1] if (i%2)==0 else [i+1,j]
        return out


    def utility(self, moves, maximizing_player=True):

        if self.cat_i==0 or self.cat_i==self.size-1 or self.cat_j==0 or self.cat_j==self.size-1:
            return float(100)

        if len(moves)==0:
            return float(-100)

        return self.eval_fn.score_proximity(self,maximizing_player)

    def apply_move(self,move,maximizing_player):
            if self.tiles[move[0]][move[1]] != 0: 
              raise InvalidMove("Invalid Move!")

            if maximizing_player:
                self.tiles[move[0]][move[1]] = 1  
            else:
                self.tiles[move[0]][move[1]] = 6        # place cat 
                self.tiles[self.cat_i][self.cat_j] = 0  # remove old cat
                self.cat_i = move[0]
                self.cat_j = move[1]

                    
    def max_Value(self, upper_game, move, maximizing_player, depth, maxdepth):
        if self.time_left()<5:
            self.terminated=True
            return [-1,-1],0
        game=copy.deepcopy(upper_game)
        if(move!=[-1,-1]):
            maximizing_player=not(maximizing_player)
            game.apply_move(move,maximizing_player)
        
        legal_moves = game.valid_moves() #["W","E","SE","SW","NE","NW"]
        if len(legal_moves)==0 or (depth==maxdepth):
            if (depth==maxdepth):
              self.reached_maxdepth = True  
            return [self.cat_i,self.cat_j], (game.size**2 - depth) * game.utility(legal_moves,maximizing_player)
        v=float("-inf")
        vtemp=v
        best_move=game.target(game.cat_i,game.cat_j,legal_moves[0])
        for s in legal_moves:
            s_pos=game.target(game.cat_i,game.cat_j,s)
            
            vtemp=max(v,self.min_Value(game,s_pos,maximizing_player,depth+1,maxdepth))
            
            if self.terminated:
                return [-1,-1],0
            if v<vtemp:
                v=vtemp
                best_move=s_pos    
  
        return best_move,v

    def min_Value(self, upper_game, move, maximizing_player, depth, maxdepth):
        if self.time_left()<5:
            self.terminated=True
            return 0
        game=copy.deepcopy(upper_game)
        maximizing_player=not(maximizing_player)
        game.apply_move(move,maximizing_player)

        #legal_moves = game.valid_moves()  # cat just moved, so he hasn't lost.
        
        if (depth==maxdepth) or\
           (game.cat_i==0 or game.cat_i==self.size-1 or game.cat_j==0 or game.cat_j==self.size-1):
            if (depth==maxdepth):
                self.reached_maxdepth = True 
            return (game.size**2 - depth) * game.utility([2,3,4],maximizing_player)
            
        v=float("inf")
        
        for i in range(game.size):      
            for j in range(game.size): 
                if game.tiles[i][j]!=0: 
                    continue            
                s = [i,j]

                placeholder,temp = self.max_Value(game,s,maximizing_player,depth+1,maxdepth)

                v = min(v,temp)
                if self.terminated:
                    return 0
        return v

    def minimax(self, max_depth=float("inf"), maximizing_player=True):
        best_move, best_val = self.max_Value(self,[-1,-1],maximizing_player,0,max_depth)
        return best_move, best_val

    def time_left(self):
        return  (self.deadline - time.time()) * 1000

    def print_tiles(self):
        i=0
        while i < self.size: 
            print(self.tiles[i])
            if i+1 < self.size:
                print("",self.tiles[i+1])
            i=i+2
        return



    def ab_max_Value(self, upper_game, move, alpha, beta, maximizing_player, depth, maxdepth):
        if self.time_left()<5:
            self.terminated=True
            return [-1,-1],0
        game=copy.deepcopy(upper_game)
        if(move!=[-1,-1]):
            maximizing_player=not(maximizing_player)
            game.apply_move(move,maximizing_player)

        legal_moves = game.valid_moves() #["W","E","SE","SW","NE","NW"]
        if len(legal_moves)==0 or (depth==maxdepth):
            if (depth==maxdepth):
                self.reached_maxdepth = True 
            return [self.cat_i,self.cat_j], (game.size**2 - depth) * game.utility(legal_moves,maximizing_player)
        v=float("-inf")
        vtemp=v
        best_move=game.target(game.cat_i,game.cat_j,legal_moves[0])
        for s in legal_moves:
            s_pos=game.target(game.cat_i,game.cat_j,s)

            vtemp=max(v,self.ab_min_Value(game,s_pos,alpha,beta,maximizing_player,depth+1,maxdepth))
            
            if self.terminated:
                return [-1,-1],0
            if v<vtemp:
                v=vtemp
                best_move=s_pos
            if v>=beta:
                return best_move,v
            alpha=max(alpha,v) 
        return best_move,v

    def ab_min_Value(self, upper_game, move, alpha, beta, maximizing_player, depth, maxdepth):
        if self.time_left()<5:
            self.terminated=True
            return 0
        game=copy.deepcopy(upper_game)
        maximizing_player=not(maximizing_player)
        game.apply_move(move,maximizing_player)

        if (depth==maxdepth) or\
           (game.cat_i==0 or game.cat_i==self.size-1 or game.cat_j==0 or game.cat_j==self.size-1):
            if (depth==maxdepth):
                self.reached_maxdepth = True 
            return (game.size**2 - depth) * game.utility([2,3,4],maximizing_player)

        v=float("inf")   
             
        for i in range(game.size):      
            for j in range(game.size):  
                if game.tiles[i][j]!=0: 
                    continue            
                s = [i,j]
                
                placeholder,temp=self.ab_max_Value(game,s,alpha,beta,maximizing_player,depth+1,maxdepth)
                
                v = min(v,temp)
                if self.terminated:
                    return 0

                if v<=alpha:
                    return v
                beta=min(beta,v)
        return v


    def alphabeta(self, max_depth=float("inf"), alpha=float("-inf"), beta=float("inf"), maximizing_player=True):
        best_move, best_val = self.ab_max_Value(self,[-1,-1],alpha,beta,maximizing_player,0,max_depth)
        return best_move, best_val


    def iterative_deepening(self,ab):
        self.terminated=False
        best_depth=0
        output_move, utility = [self.cat_i,self.cat_j],0
        for i in range(1,self.size**2):        
          self.reached_maxdepth = False
          best_move, utility = self.alphabeta(max_depth=i) if ab else self.minimax(max_depth=i)
          if self.terminated:
            break
          else:
            output_move = best_move
            best_depth = i
            elapsed_time = (time.time() - self.start_time) * 1000
            print ("Done with a tree of depth %d in %.3fms " % (i,elapsed_time))
            if (self.reached_maxdepth == False): 
              break
        print('Depth reached: ',best_depth)
        return output_move,utility


class CatEvalFn():
    def score_moves(self, game, maximizing_player_turn=True):
        cat_moves=game.valid_moves()
        return len(cat_moves) if maximizing_player_turn else len(cat_moves)-1

    def score_challenge(self, game, maximizing_player_turn=True):
        
        # Write your code here

    def score_challenge(self, game, maximizing_player_turn=True):
        i, j = game.cat_i, game.cat_j

        center = game.size // 2

        center_distance = abs(i - center) + abs(j - center)
        score = 100 - center_distance  # Closer to the center is better

        if i == 0 or i == game.size - 1 or j == 0 or j == game.size - 1:
        score -= 20  # Penalize for being near the edge

        return score if maximizing_player_turn else -score
        
        return 1 if maximizing_player_turn else -1

    def score_proximity(self, game, maximizing_player_turn=True):
        distances=[100,100]
        cat_moves=game.valid_moves()
        for move in cat_moves:
            dist = 0
            i,j = game.cat_i,game.cat_j
            while True:
                dist = dist + 1
                i,j = game.target(i,j,move)
                if (i<0 or i>=game.size or j<0 or j>=game.size):
                    break
                if game.tiles[i][j] != 0:
                    dist = dist*5
                    break
            distances.append(dist)

        distances.sort() 
        return game.size*2-(distances[0] if maximizing_player_turn else distances[1])
