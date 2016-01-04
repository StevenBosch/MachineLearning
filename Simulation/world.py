""""Our World module.

Currently values are set at:
Goal = 100 (G in map)
Block = 10 (B in map)
Start = 0 (S in map)
Path = -1 (. in map)
Wall = -10 (# in map)
"""
import numpy as np
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
