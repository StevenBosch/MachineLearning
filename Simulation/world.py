import numpy as np

""""
Currently values are set at:
Goal = 100 (G in map)
Block = 10 (B in map)
Start = 0 (S in map)
Path = -1 (. in map)
Wall = -10 (# in map)
"""

WorldStates = {
    "free": 0,
    "agent": 1,
    "wall": 2,
    "block": 3
}

def nextToBlock(agent, world):
	if np.absolute(agent.state[0]-world.block[0]) == 1:
		return True
	elif np.absolute(agent.state[1]-world.block[1]) == 1:
		return True
	else:
		return False

class World:
    def __init__(self, height, width, goals, walls, block, start):
        self.height = height
        self.width = width
        self.goals = goals
        self.walls = walls
        self.block = block
        self.start = start
        self.map = [
            [WorldStates["free"] for x in range(self.width)] for y in range(self.height)
        ]

    def add_objects(self):
        for i in self.walls:
            self.map[i[0]][i[1]] = WorldStates["wall"]
        for i in self.start:
            self.map[i[0]][i[1]] = WorldStates["agent"]
        self.map[self.block[0]][self.block[1]] = WorldStates["block"]

    def print_map(self):
        print("# "*(self.width+2))
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
        print("# "*(self.width+2))

    def getReward(self, state):
        pass
        
                        

