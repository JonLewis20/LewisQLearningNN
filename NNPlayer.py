# QNeural Network Q Learning Player
import numpy as np
import random
import tensorflow as tf
class NNPlayer:
    def __init__(self, learningRate , discount, explorationRate, boardHeightWidth, numGames):    
        
        #Q Learning Hyper Parameters
        self.discount = discount
        self.learningRate = learningRate
        self.discount = discount
        self.explorationRate = explorationRate
        self.explorationDelta = 1.0 / numGames    
        
        #Keep track of board dimensions for internal calculations
        self.boardHeightWidth = boardHeightWidth
        
        #Neural Network Parameters
        #Input layer will be the size of the  state board(game board x 2) since we track X e
        #ntries and O entries indifferent parts of the array
        self.inputLayerSize = 2 * boardHeightWidth * boardHeightWidth
        #Output layer will be the size of the gameboard and indicate which square to fill
        self.outputLayerSize = boardHeightWidth * boardHeightWidth
        
        #Set up tensorflow netwrok
        self.session = tf.Session()
        self.AssembleNetwork()
        self.session.run(self.initializer)
        
        
    def AssembleNetwork(self):
        #Create the neural network architecture
        
        #Define how many neurons should be in each hidden layer
        #For this exercise i chose to start testing with 2 layers 
        #starting off bigger in size than the input layer and
        #tapering down a little in the second layer
        layer1Size = self.inputLayerSize + 6
        layer2Size = self.inputLayerSize + 4
        
        #Variable to hold the input values
        self.nnInput = tf.placeholder(dtype = tf.float32, shape = [None, self.inputLayerSize])
        
        #Set up the hidden layers
        fc1 = tf.layers.dense(self.nnInput, layer1Size, activation = tf.sigmoid, kernel_initializer = tf.constant_initializer(np.zeros((self.inputLayerSize, layer1Size))))
        fc2 = tf.layers.dense(fc1, layer2Size, activation = tf.sigmoid, kernel_initializer = tf.constant_initializer(np.zeros((layer2Size, self.outputLayerSize))))
        
        self.nnOutput = tf.layers.dense(fc2, self.outputLayerSize)                     
        self.targetOutput = tf.placeholder(shape=[None, self.outputLayerSize], dtype=tf.float32)
        
        #Setup up the loss function to figure out how far we are off in accurary
        loss = tf.losses.mean_squared_error(self.targetOutput, self.nnOutput)
        
        #Setup our descent gradient optimizer
        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learningRate).minimize(loss)
                              
        self.initializer = tf.global_variables_initializer()
        
    def getQ(self, stateBoard):
    
        #One hot encode out state board so it can be put into the network
        oneHotEncodedBoard = self.OneHotEncode(stateBoard)
        
        #Run the state board through the network and return the 1D array of output values
        return self.session.run(self.nnOutput, feed_dict={self.nnInput: oneHotEncodedBoard})[0]
    
    def OneHotEncode(self, stateBoard):
        #The state board is already in one hot format, just need to reshape it into a 1D array
        reshapedBoard = stateBoard.reshape(-1)
        reshapedBoard = reshapedBoard[None, :]
        return reshapedBoard
    
    def RandomMove(self, stateBoard):
        #Returns a row, col of a random move to be made given the available squared
        splitBoard = np.split(stateBoard,2, axis=0)
        xBoard = splitBoard[0]
        oBoard = splitBoard[1]

        combinedBoard = xBoard|oBoard
        indices = np.argwhere(combinedBoard == 0)
        move = random.choice(indices)
        return move[0], move[1]
    
    def GetMove(self, stateBoard):
    #Chooses which move to make
        
        #Exploration allows the neural net to avoid getting stuck in a rut based on intial weightings
        #If rand returns a number lower than the current exploration rate, perform a random move which 
        #will help the network explore the rewards of paths it has not seen yet
        if random.random() > self.explorationRate:
            #Calculate a move based on Q learning
            
            #Grab Q array for current stateBoard
            qArray = self.getQ(stateBoard)
            
            #Determine the set of legal moves
            splitBoard = np.split(stateBoard,2, axis=0)
            xBoard = splitBoard[0]
            oBoard = splitBoard[1]
            combinedBoard = xBoard|oBoard
            indices = np.argwhere(combinedBoard == 0)
            
            #Create a list of row, col pairs that are legal moves that can be made
            flattenedIndices =  list()
            for rcSet in indices:
                flattenedIndices.append(rcSet[0] * self.boardHeightWidth + rcSet[1])
            
            #Pick the index fromt the Q table which has the highest
            #value  for the given set of valid choices. 
            #flattenedIndex will be the 1D index of the move to be made 
            flattenedIndex = flattenedIndices[np.argmax(qArray[flattenedIndices])]
 
            
            #Unflatted this index into  a row, col set
            if flattenedIndex < self.boardHeightWidth - 1:
                col = flattenedIndex
                row = 0
            else:
                row =  flattenedIndex//self.boardHeightWidth
                col = (flattenedIndex % self.boardHeightWidth)
            
            return int(row), int(col)
        
        else:
            #Return a random move
            return self.RandomMove(stateBoard)
        
    def Train(self, oldStateBoard, newStateBoard, action, reward):
    #Trains the neural network according to the Q learning methedology
    #Takes in the state of the board before making a move as well as what move was
    #made, the state of the board after that move and the reward that was given for the move
        
        #Grab Q values for each state
        oldStateQValues = self.getQ(oldStateBoard)
        newStateQValues = self.getQ(newStateBoard)
        
        #Calculate the Q value given the future state, reward and discount hyper parameter
        oldStateQValues[action] = reward + self.discount * np.amax(newStateQValues)
        
        #Set up neural net trainig data 
        trainingInput = self.OneHotEncode(oldStateBoard)
        targetOutput = [oldStateQValues]
        trainingData = {self.nnInput: trainingInput, self.targetOutput: targetOutput}
        
        #Train the model given the current moves
        self.session.run(self.optimizer, feed_dict = trainingData)

    def Update(self, oldStateBoard, newStateBoard, row, col, reward):
        #Translate 2d row col values into a 1D index for th corresponding output layer index
        flattenedIndex = row * self.boardHeightWidth + col
        
        #Train the mode
        self.Train(oldStateBoard, newStateBoard, flattenedIndex, reward)
        
        #Lower the exploration rate. Early on, the model should do more random
        #exploration to stop stagnation but as the model begins to get its feet under it
        #lower that and let the q learnin take over
        if (self.explorationRate - self.explorationDelta) >= 0:
                self.explorationRate -= self.explorationDelta
