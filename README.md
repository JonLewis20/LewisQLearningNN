# LewisQLearningNN

<h2>Objective</h2>
The objective of this project is to explore how Q learning can be adapted to help a neural network learn how to play the game of tic tac toe through reinfrocement learning. 

In the past, I have implimented neural networks where it is known if the answer is right or wrong in a single run through network (e.g. given this set of input features, what is the label of the input). For this project, I was interested in exploring how the same concepts could be applied to situations where the right answer cannot be known in the moment and instead relies on a set of moves to determine which moves lead to the right answer such as in the game of tic tac toe. 

DeepMind's AlphaGo was the inspiration for this project idea. Go is a board game with a large play area and 2 players. It is widely viewed as the most difficult game for a computer to play well at due to the seemingly infintensimal set of moves that can be made and complex nature of the game. DeepMind created their neural network player usubg reinforcement learning and was the first program to beat a professional go player.

Reinforcement learning networks have wide reaching implications far beyond playing games but a simple game like tic tac toe is the perfect starting point for exploring the concept.

While tic tac toe is a solved game, and extermely simple to learn the right move for any given countermove, it none the less will show the value of reinforcement learning using q learning.

The overarching goals are as follows:
<ol>
  <li> Create a simulation program which allows two AI players to play a game of tic tac toe on a configurable size square board</li>
  <li>Create a player who will pick random moves at each turn</li>
  <li>Create a player who will use q learning within a neural network to play against the random player</li>
  <li>Explore how the following parameter effect the win/loss/draw rates
    <ol>
      <li>Board Size: 3x3 vs 4x4 vs 5x5</li>
     </ol>
  </li>
  </ol>

<h3>How To Run</h3>
Open a terminal, change directory to Documents/LewisQLearning and enter the following without the quotations:
"python3 main.py"

<h3>Background Info</h3>
In the most basic neural networks, the network takes in a vast set of data and labels and trains based on whether the single run through the network produced the right or wrong answer. An example would be machine learning application where the network tries to determine if theres an apple in a picture. To train the network, you need to provide a large number of pictures which are all labeled as to if an apple is present or not. The network then continuously runs these through and adjusts its weights and bias' until it starts to get the answers right.

Reinforcement learning on the other hand forgoes the vast labeled data piles and instead uses a model to determine the labels. This model will "reward" the network for performing a good action or "punish it" for taking a bad action. 

Neither of these methods however work in their most basic forms when there is no instantaneous reward for a situation. In the game of tic tac toe, the reward is only provided for winning a game. The moves that lead up to that can't explicitly be given a reward since that move only matters in relationship to all the moves made.

This is where Q learning comes into play. Instead of looking at a single move, you instead look at the reward of that single move along with the reward of all future moves you could make. This is analogous to a chess player who spends the time to think where their current move puts them  in another 5 steps before making the move. In this same analogy, the chess player may have identified multiple ways to win a game and thus considers which move in the moment is most likely to lead to the ideal win case. In essence, the major reward is winning the game and by thinking backwards to the current move, the chess player is bleeding the reward backwards in time so that a win move is a huge reward but taking a step which leads closer to that win move still yeilds some level of reward. This is exactly what happens with q learning. As the network experiences more games, the reward values begin to bleed backwards in time so that the network is able to begin seeing which early game moves are most likely to lead to a win.

<h2>Process</h2>
<ol>
<li>Create a simulation envoronment in which two players can play a game of tic tac toe. This involves creation of the gameboard, methods for checking win/loss/draw conditions and a method to return a reward value for that move(In this case rewards are given for win/loss/draw) and those values are defined within the TicTacToeBoard class</li>
<li>Create a player class which only chooses random moves. This is a stand in for a real player or another AI opponent. This was kept as a random player to control scope for this project</li>
<li>Create a player class which uses q learning to make its move.
  <ol>
    <li>Accepts hyperparameters to set up the Q learning function and neural network architecture</li>
    <li>Impliments exploration so that the player initially starts by making random moves and then, over time, as the network has seen           more training samples, moves towards only making decisions through the neural networks. This prevents stagnation in training if         the neuron weightings have only seen a couple examples but become strongly biased towards a move.</li>
    <li>A training method to backpropogate q values through the network after each move</li>
  </ol>
  </li>
<li>Let the players play against eachother for a given number of game</li>
<li>Each turn of the game, when the q learning player makes a move, feed the reward for that move into the training method which feeds the previous and current states of the board and their q values back into the network to train it.</li>
</ol>

<h2>Results</h2>
Each trial was conducted with 3001 games played with win/loss/draw percentages reported out for each 300 games. As the neural network player will initially almost always randomly pick a square, we would expect intial sets of games to produce a 1:1 win ratio as would be expected by 2 players picking moves at random. Over time though as the neural network player shifts towards using what its learned, we would expect that win to loss ratio to increase.

<h3>Parameters</h3>
Each set of trials uses the following hyper parameters as the base setting. Changes to these base settings are noted in each set of tests
<ol>
  <li>Win Reward: 10</li>
  <li>Loss Reward: -10</li>
  <li>Draw Reward: 1</li>
  <li>Board Size: 3x3</li>
  <li>Q Learning Rate: .1</li>
  <li>Q Learning Discount: .9</li>
  <li>Q Learning Exploration Rate: 1.0</li>
</ol>

<h3>Board Size Test Results</h3>
3x3:
<ol>
  <li>First 300 Games</li>
  X's Won: 25%   O's Won: 51%  Draws: 24.33%
  <li>Last 300 Games</li>
  X's Won: 46%   O's Won: 32%  Draws: 22%
  <li>Overall Stats</li>  
  X's Won: 42.89%   O's Won: 38.42%  Draws: 18.69%
</ol>
4x4:
<ol>
  <li>First 300 Games</li>
  X's Won: 28%   O's Won: 28%  Draws: 44%
  <li>Last 300 Games</li>
  X's Won: 48.33%   O's Won: 22.33%  Draws: 29.33%
  <li>Overall Stats</li>  
  X's Won: 45.92%   O's Won: 22.59%  Draws: 31.49%
</ol>
5x5:
<ol>
  <li>First 300 Games</li>
  X's Won: 21%   O's Won: 21.67%  Draws: 57.67%
  <li>Last 300 Games</li>
  X's Won: 47.33%   O's Won: 15.67%  Draws: 37%
  <li>Overall Stats</li>  
  X's Won: 39.49%   O's Won: 17.53%  Draws: 42.99%
</ol>



