"""Our main module."""
import world as w
import agentExponentExploration as ag
import numpy as np
import matplotlib.pyplot as plt


def nextToBlock(y, x, world):
    """Check if the block is next to the agent."""
    if np.absolute(y-world.block[0]) + np.absolute(x-world.block[1]) == 1:
        return True
    return False


def checkAgentMove(world, y, x, action):
    """Check to see if a move is possible for an agent.

    Return true (not false) if the move is not OOB or to another object,
    or if the block is next to the agent, if he wants to grab
    """
    if (action == ag.Actions["grab"] and not nextToBlock(y, x, world)):
            return False
    elif (action == ag.Actions["left"] and
          (x == 0 or world.map[y][x-1] != w.WorldStates["free"])):
            return False
    elif ((action == ag.Actions["right"]) and
          (x == world.columns-1 or world.map[y][x+1] != w.WorldStates["free"])):
            return False
    elif ((action == ag.Actions["up"]) and
          (y == 0 or world.map[y-1][x] != w.WorldStates["free"])):
            return False
    elif ((action == ag.Actions["down"]) and
          (y == world.rows-1 or world.map[y+1][x] != w.WorldStates["free"])):
            return False
    return True


def moveAgent(agent, world):
    # Move a single agent if he can take his intended action
    y, x = agent.state
    agent.prevGrasped = agent.grasped

    if checkAgentMove(world, y, x, agent.action):
        if agent.action == ag.Actions["left"]:
            x = x - 1
        if agent.action == ag.Actions["right"]:
            x = x + 1
        if agent.action == ag.Actions["up"]:
            y = y - 1
        if agent.action == ag.Actions["down"]:
            y = y + 1
        if agent.action == ag.Actions["grab"] and agent.grasped == 0:
            agent.grasped = 1
            agent.reward = 1

        world.map[agent.state[0]][agent.state[1]] = w.WorldStates["free"]
        world.map[y][x] = w.WorldStates["agent"]
        agent.state = (y, x)


def moveBlock(agents, world):
    """Move to block according to the actions taken by the agents.

    For the action the agents want to do,
    check if it's possible for the agents and the block.
    If so,  first move the block and then let the agents move using moveAgent.
    """
    yblock = world.block[0]
    xblock = world.block[1]
    possibleMove = True

    # Set the state of the block on "free",
    # so the agents see that as movable space, this is reset later on
    world.map[world.block[0]][world.block[1]] = w.WorldStates["free"]

    for agent in agents:
        if not checkAgentMove(world, agent.state[0], agent.state[1], agent.action):
            possibleMove = False
            break

    if agents[0].action == ag.Actions["left"]:
        if xblock == 0 or world.map[yblock][xblock-1] == w.WorldStates["wall"]:
            possibleMove = False
        xblock -= 1

    if agents[0].action == ag.Actions["right"]:
        if xblock == world.columns - 1 or world.map[yblock][xblock+1] == w.WorldStates["wall"]:
            possibleMove = False
        xblock += 1

    if agents[0].action == ag.Actions["up"]:
        if yblock == 0 or world.map[yblock - 1][xblock] == w.WorldStates["wall"]:
            possibleMove = False
        yblock -= 1

    if agents[0].action == ag.Actions["down"]:
        if (
            yblock == world.rows - 1
            or world.map[yblock + 1][xblock] == w.WorldStates["wall"]
        ):
            possibleMove = False
        yblock += 1

    if possibleMove:
        world.block = (yblock, xblock)
        # Calculate
        if any(world.block == g for g in goals):
            for agent in agents:
                if agent.grasped:
                    agent.reward = 10
        [moveAgent(agent, world) for agent in agents]

    world.map[world.block[0]][world.block[1]] = w.WorldStates["block"]


def updateWorld(agents, world, steps, epoch):
    """Update everythings position in the world.

    If not all agents grabbed the block, move the single ones,
    else move the block if their actions correspond.
    """
    allGrasped = all(ag.grasped for ag in agents)
    sameAction = all(ag.action == agents[0].action for ag in agents)
    if allGrasped:
        steps[1][epoch] += 1
        if sameAction:
            moveBlock(agents, world)
    else:
        steps[0][epoch] += 1
        for agent in agents:
            if not agent.grasped:
                moveAgent(agent, world)
    return world

if __name__ == "__main__":

    # Some parameters
    numberOfAgents = 2
    tau = 0.99
    alpha = 0.1
    gamma = 0.9

    # World parameters
    rows = 5
    columns = 15
    goals = [(3, 1)]
    walls = []
    block = (0, 14)

    # Starting positions of the agents,
    # should be the same amount as the number agents
    start = [(4, 14), (0, 0)]

    # Create the agents
    agents = [ag.Agent(start[i], rows, columns)
              for i in range(numberOfAgents)]

    # Create the world and everything in it
    new_world = w.World(rows, columns, goals, walls, block, start)
    new_world.add_objects()
    new_world.print_map()

    epochs = 2000
    steps = np.zeros((2,epochs))
    for epoch in range(epochs):
        print(epoch)
        # Set the agents to their starting positions
        for index, agent in enumerate(agents):
            agent.state = start[index]
            agent.grasped = 0

        # Create the world and everything in it
        new_world = w.World(rows, columns, goals, walls, block, start)
        new_world.add_objects()

        while not any([new_world.block == g for g in goals]):
            # Save the current state
            for agent in agents:
                agent.prevState = agent.state
            # Choose an action for every agent
            [agent.chooseAction(tau, new_world) for agent in agents]
            # Perform the chosen actions (and obtain rewards)
            new_world = updateWorld(agents, new_world, steps, epoch)
            # Update the Q-values of the agents
            [agent.updateQ(alpha, gamma) for agent in agents]

            new_world.time += 1

        tau *= 0.999

    for index, agent in enumerate(agents):
        print("Agent: ", index)
        agent.print_policy(new_world)

    print(tau)

    #print(steps)
    plt.figure(1)
    plt.plot(range(epochs), steps[0], 'r-', range(epochs), steps[1], 'b-')
    plt.title('Steps per epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Steps')
    plt.legend(['Not grasped', 'Grasped'])

    plt.figure(2)
    plt.plot(range(epochs), np.convolve(steps[0], np.ones(3)/3, 'same'), 'r-', range(epochs), np.convolve(steps[1], np.ones(3)/3, 'same'), 'b-')
    plt.title('Steps per epoch (smoothed for window = 3)')
    plt.xlabel('Epoch')
    plt.ylabel('Steps')
    plt.legend(['Not grasped', 'Grasped'])

    plt.show()
