# Lights Out
## Description
Lights Out is a game played on a 5x5 board. Each cell on the board is a button as well as a light. 
When a button in a cell is toggled, the light in that cell and the lights in all cells that share an edge
are toggled. The player of the game is presented with a pattern of lights. The player's objective is to
click cells such that all of the lights on the board are turned to the off state.


## Mathematics
### All Ones Problem
A simpler objective to the problem set before the player in Lights Out is the All Ones Problem.
Here is how the problem is usually phrased (from [this](https://drive.google.com/file/d/1fjlUMWDdL4Une6CjCAvtjzAqaGQ1EMJr/view?usp=sharing) paper):

A janitor is walking through a museum in the morning, switching on the light in every room. In every room 
there is a button, but pressing this button toggles the light on/off not only in the same room but also in 
all the neighboring rooms! Can the janitor light up the entire museum?

This is called the "All Ones Problem" because the rooms can be represented as being either a "1" or a
"0" for "on" or "off". This is how the boards will be represented in programming.

Klaus Sutner gave an equivalent version of the problem and proved that the janitor can light every room if
the museum rooms create a rectangluar grid in [this](https://drive.google.com/file/d/1JBuMVqju5-RSAoG0qzMPdVo-wfO7EUs9/view?usp=sharing) 1989 paper.

The all ones problem doesn't consider other starting configurations of lights other than all off 
(which is equivalent to all on for Lights Out players).

### "Lights Out Problem"
The extensions to the All Ones Problem that are of interest to Lights Out players are these:
#### For an NxN board...
1. All all light patterns solveable? How do we know? For which N?
2. How can one find optimal solutions for a given board? 
3. What is the minimum number of clicks needed to optimally solve any solveable NxN starting configuration?

Todd Fiel and Marlow Anderson helped to answer some of these questions in their [paper](https://drive.google.com/file/d/0B48pWfermjqTWVdfbDV3SzlrRGc/view?usp=sharing). They studied the 5x5 version, 
but they gave a framework for working with bigger boards.
