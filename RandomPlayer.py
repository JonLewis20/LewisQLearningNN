import numpy as np
import random

class RandomPlayer:
#Define the random player
#This player will only make random moves on the board

    def MakeMove(self, combinedStateBoard):  
    #Decides which squar it should fill at random given what is not
    #already occupied. Return the row, col to be played
       
        #Find empty indices
        indices = np.argwhere(combinedStateBoard == 0)
        move = random.choice(indices)
        return move[0], move[1]
 
