# LewisQLearningNN

<h2>Objective</h2>
The objective of this project is to explore how Q learning can be adapted to help a neural network learn how to play the game of tic tac toe through reinfrocement learning.

<h3>Background Info</h3>
In the most basic neural networks, the network takes in a vast set of data and labels and trains based on whether the single run through the network produced the right or wrong answer. An example would be machine learning application where the network tries to determine if theres an apple in a picture. To train the network, you need to provide a large number of pictures which are all labeled as to if an apple is present or not.

Reinforcement learning on the other hand forgoes the vast labeled data piles and instead uses a model to determine the labels. This model will "reward" the network for performing a good action or "punish it" for taking a bad action. 

Neither of these methods however work in their most basic forms when there is no instantaneous reward for a situation. In the game of tic tac toe, the reward is only provided for winning a game. The moves that lead up to that can't explicitly be given a reward since that move only matters in relationship to all the moves made.

This is where Q learning comes into play. Instead of looking at a single move, you instead look at the reward of that single move along with the reward of all future moves you could make. This is analogous to a chess player who spends the time to think where their current move puts them  in another 5 steps before making the move. In this same analogy, the chess player may have identified multiple ways to win a game and thus considers which move in the moment is most likely to lead to the ideal win case. In essence, the major reward is winning the game and by thinking backwards to the current move, the chess player is bleeding the reward backwards in time so that a win move is a huge reward but taking a step which leads closer to that win move still yeilds some reward. This is exactly what happens with q learning. As the network experiences more games, the reward values begin to bleed backwards in time so that the network is able to begin seeing which early game moves are most likely to lead to a win.


Q = Reward + Discount * Future Reward








<h2>Process</h2>
<ol>
<li>Create a simulation envoronment in which two players can play a game of tic tac toe. This involves creation of the gameboard, methods for checking win/loss/draw conditions and a method to return a reward value for that move(In this case rewards are given for win/loss/draw)</li>
<li>Create a player class which only chooses random moves. This is a stand in for a real player or another AI opponent. This was kept as a random player to control scope for this project</li>
<li>Create a player class which uses q learning to make its move.
  <ol>
    <li>Accepts hyperparameters to set up the Q learning function and neural network architecture</li>
    <li>Impliment exploration so that the player initially starts by making random moves and then, over time, as the network has seen           more training samples, move towards only making decisions through the neural networks. This prevents stagnation in training if            the neuron weightings have only seen a couple examples but become strongly biased towards a move.</li>
    <li>A training method to backpropogate q values through the network after each move</li>
  </ol>
  </li>
<li>Let the players play against eachother for a given number of game</li>
<li>Each turn of the game, when the q learning player makes a move, feed the reward for that move into the training method which feeds the previous and current states of the board and their q values back into the network to train it.</li>
</ol>

<h2>Results</h2>
