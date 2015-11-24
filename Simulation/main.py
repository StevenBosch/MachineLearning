import world


def make_world():
    # The size of the world
    height = 10
    width = 10

    # Add the coordinates of the objects in the world
    goals = [(0, 9), (9, 0)]
    walls = [(2, 0), (2, 1), (2, 2), (2, 3), (2, 4)]
    block = (5, 5)
    start = [(0,0),(9,9)]

    #Create a new world
    new_world = world.World(height, width, goals, walls, block, start)
    new_world.add_objects()
    new_world.print_map()

    return new_world

new_world = make_world()