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

    #world.print_map()
    #print("States: ", agents[0].state)
    #print("Block: ", world.block)
    #print("Actions: ", agents[0].valueToActionList(agents[0].action))
    #print("Grasped: ", agents[0].grasped)
    #print("allGrasped is ", allGrasped)

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
    if len(sys.argv) != 2:
        sys.exit("""
            Please execute this script with one argument \n
            for the action selection style, i.e.\n
                "python3 main.py greedy" or \n
                "python3 main.py exponent"
        """)
    else:
        actionStyle = str(sys.argv[1])
        print("You've selection action selection style: ", actionStyle)

    # Some parameters
    epsilon = 0.9
    startTau = 0.99
    tau = startTau
    alpha = 0.1
    gamma = 0.9

    # World parameters
    rows = 5
    columns = 5
    goals = [(3, 1)]
    walls = []
    block = (0, 4)

    # Starting positions of the agents,
    # should be the same amount as the number agents
    start = [(4, 4), (0, 0)]
    numberOfAgents = len(start)

    # Create the agents
    #agents = [ag.Agent(start[i], rows, columns)
    #          for i in range(numberOfAgents)]
    agents = [ag.Agent(start, rows, columns, len(start))]

    # Create the world and everything in it
    world = w.World(rows, columns, goals, walls, block, start)
    world.add_objects()
    world.print_map()

    # Simulation settings
    epochs = 1000
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
            # Save the current state
            for agent in agents:
                agent.prevState = agent.state

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

    plt.figure(2)
    smooth = math.ceil(epochs * 0.01)
    plt.plot(
        range(epochs),
        np.convolve(steps[0], np.ones(smooth)/smooth, 'same'),
        'r-',
        range(epochs),
        np.convolve(steps[1], np.ones(smooth)/smooth, 'same'),
        'b-'


    )
    plt.title('Steps per epoch (smoothed for window = %s)'%smooth)
    plt.xlabel('Epoch')
    plt.ylabel('Steps')
    plt.legend(['Not grasped', 'Grasped'])

    plt.show()
