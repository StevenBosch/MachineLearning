""""Our World module.

Currently values are set at:
Goal = 100 (G in map)
Block = 10 (B in map)
Start = 0 (S in map)
Path = -1 (. in map)
Wall = -10 (# in map)
"""
import numpy as np
import agent as ag
""" The possible states a world cell can have."""
WorldStates = {
    "free": 0,
    "agent": 1,
    "wall": 2,
    "block": 3
}


class World:
    def __init__(self, rows, columns, goals, walls, block, start):
        """Initialize a new world with the given parameters."""
        self.rows = rows
        self.columns = columns
        self.goals = goals
        self.walls = walls
        self.block = block
        self.start = start
        # Numpy matrix is inde/xed by [row, column]
        self.map = np.zeros((rows, columns))
        self.time = 0

    def add_objects(self):
        """Add the objects such as walls to the map."""
        for i in self.walls:
            self.map[i[0]][i[1]] = WorldStates["wall"]
        for i in self.start:
            self.map[i[0]][i[1]] = WorldStates["agent"]
        self.map[self.block[0]][self.block[1]] = WorldStates["block"]

    def print_map(self):
        """Print the current situation of the map."""
        print("# "*(self.columns+2))
        for x in self.map:
            print("#", end=" ")
            for y in x:
                if y == 0:
                    print(".", end=" ")
                elif y == 1:
                    print("A", end=" ")
                elif y == 2:
                    print("#", end=" ")
                elif y == 3:
                    print("B", end=" ")
                elif y == 100:
                    print("G", end=" ")
            print("#")
        print("# "*(self.columns+2))

    def nextToBlock(self, y, x):
        """Check if the block is next to the agent."""
        if np.absolute(y-self.block[0]) + np.absolute(x-self.block[1]) == 1:
            return True
        return False

    def checkMove(self, y, x, action):
        """Check to see if a move is possible.

        Return true if the move is not OOB or to another object,
        or if the block is next to the agent, if he wants to grab
        """
        nCol = self.columns
        nRow = self.rows

        if (action == ag.Actions["grab"] and not self.nextToBlock(y, x)):
                return False
        elif (action == ag.Actions["left"] and
              (x == 0 or self.map[y][x-1] != WorldStates["free"])):
                return False
        elif ((action == ag.Actions["right"]) and
              (x == nCol-1 or self.map[y][x+1] != WorldStates["free"])):
                return False
        elif ((action == ag.Actions["up"]) and
              (y == 0 or self.map[y-1][x] != WorldStates["free"])):
                return False
        elif ((action == ag.Actions["down"]) and
              (y == nRow-1 or self.map[y+1][x] != WorldStates["free"])):
                return False
        return True

    def moveBlock(self, agents):
        """Move to block according to the actions taken by the agents.

        For the action the agents want to do,
        check if it's possible for the agents and the block.
        If so,first move the block and then move the agents using moveAgent.
        """
        yblock = self.block[0]
        xblock = self.block[1]
        possibleMove = False
        
        for a in agents:
            actionList = a.valueToActionList(a.action)
            # Set the state of the block on "free",
            # so the agents see that as movable space, this is reset later on
            self.map[self.block[0]][self.block[1]] = WorldStates["free"]

            # Check if the every agents' move is possible
            possibleMove = all(
                self.checkMove(a.state[2*i], a.state[2*i+1], actionList[i]) for i in range(a.nAgents)
            )

            # Check if the blocks' move is possible
            # we temporary have to set the agents position to free to check
            for a in agents:
                self.map[a.state[0]][a.state[1]] = WorldStates["free"]
            if possibleMove:
                possibleMove = self.checkMove(yblock, xblock, agents[0].action)
            for a in agents:
                self.map[a.state[0]][a.state[1]] = WorldStates["agent"]

            # Time to move that block
            if possibleMove:
                if actionList[0] == ag.Actions["left"]:
                    xblock -= 1
                if actionList[0] == ag.Actions["right"]:
                    xblock += 1
                if actionList[0] == ag.Actions["up"]:
                    yblock -= 1
                if actionList[0] == ag.Actions["down"]:
                    yblock += 1
                self.block = (yblock, xblock)
                # Check to see if we reached the goal
                if any(self.block == g for g in self.goals):
                    for agent in agents:
                        if agent.grasped:
                            agent.reward += 10
                # Move the agents
                [agent.moveAgent(self) for agent in agents]

            # Set the block's position to occupied on the map
            self.map[self.block[0]][self.block[1]] = WorldStates["block"]
