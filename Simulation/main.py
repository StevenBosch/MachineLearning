import world
import agent

def updateMovements(agent1, agent2, world):
    pass

if __name__ == "__main__":
    height = 10
    width = 10
    tau = 0.99

    goals = [(0, 9), (9, 0)]
    walls = [(2, 0), (2, 1), (2, 2), (2, 3), (2, 4)]
    block = (5, 5)
    start = [(0, 0), (9, 9)]

    new_world = world.World(height, width, goals, walls, block, start)
    new_world.add_objects()
    new_world.print_map()

    agent1 = agent.Agent(height*width, start[0], height, width)
    agent2 = agent.Agent(height*width, start[1], height, width)

    for x in range(1000):
        agent1.chooseAction(tau)
        agent2.chooseAction(tau)

        updateMovements(agent1, agent2, world)

        agent1.updateQ()
        agent2.updateQ()

        tau = tau * 0.999
