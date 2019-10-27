import numpy as np
import random
import tensorflow as tf
class TicTacToeBoard:
    def __init__(self, boardHeightWidth):
        #Hyper-parameters to be used with the neural network agent
        #Defines the rewards to be used i nthe Q function
        self.winReward = 10
        self.loseReward = -10
        self.drawReward = 1;
        self.moveReward = 0;
        self.illegalMoveReward = -1
        
        #The tic tac toe board will always be square the have a 1D lenght of this
        self.boardHeightWidth = boardHeightWidth
        
        #Define a blank game state board. The board must always be square. The stateBoard however
        #will have 2 boards contained it is as defined by an M x 2N array.In a regular 3 row/col
        #tic tac toe board this would mean there will be a 3 x 6 array. The first 3 rows containig
        #entries for the X player and the last 3 rows containing entries for the O player.
        #The initial state of the board will be all zeros
        self.stateBoard = np.zeros([boardHeightWidth * 2, boardHeightWidth], dtype=int)
        
    def CheckWinCondition(self):
        #Given the current state of the game board, checks  to see if a winning move has been 		#made    
        #Initialize winner. -1: No winner, 0: X wins, 1: O wins

        #Check if 3 in a row horizontally    
        for row in range(self.boardHeightWidth):
            #Reset counters
            xCounter = 0;
            oCounter = 0;
            for col in range(self.boardHeightWidth):
                if self.stateBoard[row][col] == 1:
                    xCounter += 1
                if self.stateBoard[row + self.boardHeightWidth][col] == 1:
                    oCounter += 1
            if xCounter == self.boardHeightWidth:
                winner = 0
                #display("Winner by Horizonal")
                return winner
            elif oCounter == self.boardHeightWidth:
                winner = 1
                #display("Winner by Horizontal")
                return winner

        #Check if 3 in a row vertically
        for col in range(self.boardHeightWidth):
            #Reset counters
            xCounter = 0;
            oCounter = 0;
            for row in range(self.boardHeightWidth):
                if self.stateBoard[row][col] == 1:
                    xCounter += 1
                if self.stateBoard[row + self.boardHeightWidth][col] == 1:
                    oCounter += 1
            if xCounter == self.boardHeightWidth:
                winner = 0
                return winner
            elif oCounter == self.boardHeightWidth:
                winner = 1
                return winner
 

        #Check left to right diagonal
        for colrow in range(self.boardHeightWidth):
            #Reset counter 
            xCounter = 0
            oCounter = 0
            
            if self.stateBoard[colrow][colrow] == 1:
                 xCounter += 1
            if self.stateBoard[colrow + self.boardHeightWidth][colrow] == 1:
                 oCounter += 1                         
        if xCounter == self.boardHeightWidth:
            winner = 1
            return winner
        if oCounter == self.boardHeightWidth:
            winner = 0
            return winner
        
        
        #Reset counter 
        xCounter = 0
        oCounter = 0
        
        #Check right to left diagonal
        colList = np.arange(self.boardHeightWidth-1, -1, -1).tolist()
        for row in range(self.boardHeightWidth):
            col = colList[row]
            if self.stateBoard[row][col] == 1:
                xCounter += 1
            if self.stateBoard[row + self.boardHeightWidth][col] == 1:
                oCounter += 1
        if xCounter == self.boardHeightWidth:
            winner = 1
            return winner
        if oCounter == self.boardHeightWidth:
            winner = 0
            return winner
        
        #If we make it here, there is no winner
        return -1
    
    def CheckDrawCondition(self):
        #Check to see if the board is completely full
        
        #Grab the compressed board which maries the X and O's state board
        compressedBoard = self.CompressedBoard()
        
        #If the number of ones contained in the compressed board is equal to the number of squares
        #on the game board then its a draw
        if np.count_nonzero(compressedBoard) == (self.boardHeightWidth * self.boardHeightWidth):      
            return True
        else:
            return False
    
    def MakeMove(self, row, col, playerNum):
        #Each agent playing the game will call this function to make their move
        #during their turn by entering the row and col they want to mark on th game board
        #playerNum 0 = X, 1 = O
        
        #Check if the player is trying to place a mark in a column that is already
        #occupied by a mark
        if self.CheckIllegalMove(row, col) == True:
            display("Move by player: " + str(playerNum) + " was illegal")
            return self.illegalMoveReward
        
        #Place their mark
        if playerNum == 0:
            self.stateBoard[row][col] = 1
        elif playerNum == 1:
            row = row + self.boardHeightWidth
            self.stateBoard[row][col] = 1
            
        #Check if the move resulted in a game ending condition    
        #Return the proper award if so
        winner = self.CheckWinCondition()
        if winner == 0 and playerNum == 0:
            return self.winReward
        elif winner == 1 and playerNum == 1:
            return self.winReward
        elif self.CheckDrawCondition() == True:
            return self.drawReward
        else:
            return self.moveReward
    
    def CheckIllegalMove(self, row, col):
        #sChecks if the player is trying to place a mark in a column that is already
        #occupied by a mark
        #Returns a boolean value indicating if the move is illegal
        
        #Split stateBoard into player components
        splitBoard = np.split(self.stateBoard,2, axis=0)
        xBoard = splitBoard[0]
        oBoard = splitBoard[1]
        combinedBoard = xBoard|oBoard
        
        if combinedBoard[row][col] == 1:
            return True
        else:
            return False
        
    def CompressedBoard(self):
    #Returns a stateBoard which has been collapsed down to a game board to show which spaces have something
    #in them and which are empty. Takes a M x 2N array, splits it in half along the N axis and bitwise ors
    #the two into an MxN array
        splitBoard = np.split(self.stateBoard,2, axis=0)
        xBoard = splitBoard[0]
        oBoard = splitBoard[1]
        combinedBoard = xBoard|oBoard
        return combinedBoard
    
    def ResetBoard(self):
        #Resets the state board at the end of each game
        self.stateBoard = np.zeros([self.boardHeightWidth * 2, self.boardHeightWidth], dtype=int)
