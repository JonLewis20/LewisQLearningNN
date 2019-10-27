#Main Program
from TicTacToeBoard import TicTacToeBoard
from RandomPlayer import RandomPlayer
from NNPlayer import NNPlayer
import numpy as np

if __name__ == '__main__':
	#Q Learning Hyper-parameters
	learningRate = .1
	discount = .9
	explorationRate = 1.0

	#Number of games to train on
	numGames = 1200

	#Report out stastics of games won/lost/draw after this many games
	reportingPeriod = 300

	#Board dimensions
	boardHeightWidth = 4

	#Setup game board
	GameBoard = TicTacToeBoard(boardHeightWidth)
	gameOver = False

	#Set up our players
	RandomAgent = RandomPlayer()
	NNAgent = NNPlayer(learningRate , discount, explorationRate, boardHeightWidth, numGames)

	#Counters for keeping track of game statistics
	overallWinCounter = np.array([0, 0])
	subsectionWinCounter = np.array([0, 0])
	overallDrawCounter = 0;
	subsectionDrawCounter = 0;
	
	#Give same space between the intialization text and results
	print("\n \n \n")
	for gameNum in range (numGames):
		#Reset Game Board
		GameBoard.ResetBoard()

		while True:		    
		    #X makes move
		    board = GameBoard.stateBoard
		    oldStateBoard =  GameBoard.stateBoard
		    combinedBoard = GameBoard.CompressedBoard()
		    row, col = RandomAgent.MakeMove(combinedBoard)
		    reward = GameBoard.MakeMove(row, col, 1)
		    newStateBoard = GameBoard.stateBoard

		    winner = GameBoard.CheckWinCondition()
		    draw = GameBoard.CheckDrawCondition()
		    if winner != -1:
		        #print("We have a winner: Player " + str(winner))
		        #display("Game # " + str(gameNum))
		        #display(board)
		        subsectionWinCounter[winner] += 1
		        #If the opponent wins, the model needs to train itself on the loss
		        #Gives the negative of the reward the opponent got
		        NNAgent.Update(oldStateBoard, newStateBoard, nnRow, nnCol, -reward)
		        break
		    elif draw == True:
		        #Game ended in draw
		        #print("Game ended in draw")
		        subsectionDrawCounter += 1
		        break   
		    
		    #O makes move
		    oldStateBoard =  GameBoard.stateBoard
		    nnRow, nnCol = NNAgent.GetMove(GameBoard.stateBoard)
		    reward = GameBoard.MakeMove(nnRow, nnCol, 0)
		    newStateBoard = GameBoard.stateBoard
		    NNAgent.Update(oldStateBoard, newStateBoard, nnRow, nnCol, reward)
		    
		    winner = GameBoard.CheckWinCondition()
		    draw = GameBoard.CheckDrawCondition()
		    if winner != -1:
		        #print("We have a winner: Player " + str(winner))
		        #display("Game # " + str(gameNum))
		        #isplay(board)
		        subsectionWinCounter[winner] += 1
		        break
		    elif draw == True:
		        #print("Game ended in draw")
		        subsectionDrawCounter += 1
		        break   
		    
		#Report out game statistics per the reporting period
		if gameNum % reportingPeriod == 0 and gameNum != 0:
		    overallWinCounter[0] += subsectionWinCounter[0]
		    overallWinCounter[1] += subsectionWinCounter[1] 
		    overallDrawCounter += subsectionDrawCounter
		    
		    xPercentage = round((subsectionWinCounter[0]/reportingPeriod)*100,2)
		    oPercentage = round((subsectionWinCounter[1]/reportingPeriod)*100,2)
		    drawPercentage = round((subsectionDrawCounter/reportingPeriod)*100,2)	    		
		    print("Stats for the past " + str(reportingPeriod) + " games")
		    print("X's won: " + str(subsectionWinCounter[0]) + "  O's won: " + str(subsectionWinCounter[1]) + "  Draws:" + str(subsectionDrawCounter)) 
		    print("Percentage X's won: " + str(xPercentage) + "%  Percentage O's won: " + str(oPercentage) + "%  Percentage Draws:" + str(drawPercentage) + "% \n" ) 

		    
		    #Reset the reporting period specific counters
		    subsectionWinCounter[0] = 0
		    subsectionWinCounter[1] = 0
		    subsectionDrawCounter = 0
		    
	#Report out overall stats across all games
	xPercentage = round((overallWinCounter[0]/numGames)*100,2)
	oPercentage = round((overallWinCounter[1]/numGames)*100,2)
	drawPercentage = round((overallDrawCounter/numGames)*100,2)
	print("Stats for the overal run of games")
	print("Percentage X's won: " + str(xPercentage) + "%  Percentage O's won: " + str(oPercentage) + "%  Percentage Draws:" + str(drawPercentage) + "%")     
	print("X's won: " + str(overallWinCounter[0]) + "  O's won: " + str(overallWinCounter[1]) + "  Draws:" + str(overallDrawCounter))

														

