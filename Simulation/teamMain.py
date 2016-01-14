"""Our main module."""
import sys
import teamWorld as w
import teamAgent as ag
import numpy as np
import matplotlib.pyplot as plt
import math


def updateWorld(agents, world, steps, epoch):
    """Update everythings position in the world.

    If not all agents grabbed the block, move the single ones,
    else move the block if their actions correspond.
    """
    # Use for team agent
    agents[0].allGrasped = all(agents[0].grasped)
    sameAction = len(np.unique(
        agents[0].valueToActionList(agents[0].action)
        )) == 1

    if agents[0].allGrasped:
        steps[1][epoch] += 1
        if sameAction:
            world.moveBlock(agents)
    else:
        steps[0][epoch] += 1
        for agent in agents:
            agent.moveAgent(world)
    return world

if __name__ == "__main__":

    # Parse the cla
    if len(sys.argv) != 3:
        sys.exit("""
            Please execute this script with two arguments \n
            for the action selection style, i.e.\n
                "python3 main.py greedy complex" or \n
                "python3 main.py exponent simple"
        """)
    else:
        actionStyle = str(sys.argv[1])
        environment = str(sys.argv[2])
        print("You've selection action selection style: ", actionStyle)

    # Some parameters
    epsilon = 0.9
    startTau = 0.99
    tau = startTau
    alpha = 0.1
    gamma = 0.9

    # World parameters
    if environment == "complex":
        rows = 8
        columns = 8
        goals = [(6, 6)]
        walls = [(2,3),(3,3), (3,4), (3,5), (0,3),(0,4),(4,4),(5,4),(0,1),(1,1),(3,1),(4,1),(5,1),(6,1),(7,1)]
        block = (0, 5)
        start = [(0, 0), (4, 0)]
        numberOfAgents = len(start)
    else:
        rows = 5
        columns = 5
        goals = [(3, 1)]
        walls = []
        block = (0,4)
        start = [(0, 0), (3, 3)]
        numberOfAgents = len(start)

    # Create the agents
    agents = [ag.Agent(start, rows, columns, len(start))]

    # Create the world and everything in it
    world = w.World(rows, columns, goals, walls, block, start)
    world.add_objects()
    world.print_map()

    # Simulation settings
    epochs = 2000
    steps = np.zeros((2, epochs))

    for epoch in range(epochs):
        print(epoch)
        
        # Set the agents to their starting positions
        for agent in agents:
            agent.reset() 

        # Create the world and everything in it
        world = w.World(rows, columns, goals, walls, block, start)
        world.add_objects()
        while not any([world.block == g for g in goals]):
            # Save the previous state and grasped (this has to be done per element, else it becomes a pointer)
            for agent in agents:
                for number in range(len(start)):
                    agent.prevState[number*2] = agent.state[number*2]
                    agent.prevState[number*2+1] = agent.state[number*2+1]
                    agent.prevGrasped[number] = agent.grasped[number]

            # Choose an action for every agent
            if actionStyle == "greedy":
                [agent.chooseGreedyAction(epsilon, world) for agent in agents]
            else:
                [agent.chooseAction(tau, world) for agent in agents]

            # Perform the chosen actions (and obtain rewards)
            world = updateWorld(agents, world, steps, epoch)
            # Update the Q-values of the agents
            [agent.updateQ(alpha, gamma) for agent in agents]
           
            if steps[0][epoch] > 10000 or steps[1][epoch] > 10000:
                print("Took too long")
        
        tau -= (startTau-0.1) / epochs
    
        
    plt.figure(1)
    plt.plot(range(epochs), steps[0], 'r-', range(epochs), steps[1], 'b-')
    plt.title('Steps per epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Steps')
    plt.legend(['Not grasped', 'Grasped'])
    plt.draw()
    plt.savefig('TeamQ.png')
    
    plt.figure(2)
    smooth = math.ceil(epochs * 0.1)
    plt.plot(
        range(epochs-smooth+1),
        np.convolve(steps[0], np.ones(smooth)/smooth, 'valid'),
        'r-',
        range(epochs-smooth+1),
        np.convolve(steps[1], np.ones(smooth)/smooth, 'valid'),
        'b-'
    )
    plt.title('Steps per epoch (smoothed for window = %s)'%smooth)
    plt.xlabel('Epoch')
    plt.ylabel('Steps')
    plt.legend(['Not grasped', 'Grasped'])
    plt.draw()
    plt.savefig('TeamQ_smoothed.png')