import numpy

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
    "block": 3,
    "goal": 4
}

class World:
    def __init__(self, height, width, goals, walls, block, start):
        self.height = height
        self.width = width
        self.goals = goals
        self.walls = walls
        self.block = block
        self.start = start
        self.map = [
            [Worldstates.free.value for x in range(self.width)] for y in range(self.height)
        ]

    def add_objects(self):
        for i in self.goals:
            self.map[i[0]][i[1]] = WorldStates.goal.value
        for i in self.walls:
            self.map[i[0]][i[1]] = WorldStates.wall.value
        for i in self.start:
            self.map[i[0]][i[1]] = WorldStates.agent.value
        self.map[self.block[0]][self.block[1]] = Worldstates.block.value

    def print_map(self):
        print("# "*(self.width+2))
        for x in self.map:
            print("#", end=" ")
            for y in x:
                if y == -10:
                    print("#", end=" ")
                elif y == -1:
                    print(".", end=" ")
                elif y == 0:
                    print("S", end=" ")
                elif y == 10:
                    print("B", end=" ")
                elif y == 100:
                    print("G", end=" ")
            print("#")
        print("# "*(self.width+2))

    def getReward(self, state):
        pass
