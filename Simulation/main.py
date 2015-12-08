"""Our main module."""
import world as w
import agent as ag
import numpy as np

def nextToBlock(x, y, world):
    """Check if the block is next to the agent."""
    if np.absolute(x-world.block[0]) + np.absolute(y-world.block[1]) == 1:
        return True
    return False


def checkAgentMove(world, x, y, action):
    """Check to see if a move is possible for an agent.

    Return true (not false) if the move is not OOB or to another object,
    or if the block is next to the agent, if he wants to grab
    """
    if (action == ag.Actions["grab"] and not nextToBlock(x, y, world)):
            return False
    elif (action == ag.Actions["left"] and
          (x == 0 or world.map[x+1][y] != w.WorldStates["free"])):
            return False
    elif ((action == ag.Actions["right"]) and
          (x == world.width-1 or world.map[x+1][y] != w.WorldStates["free"])):
            return False
    elif ((action == ag.Actions["up"]) and
          (y == 0 or world.map[x][y-1] != w.WorldStates["free"])):
            return False
    elif ((action == ag.Actions["down"]) and
          (y == world.height-1 or world.map[x][y+1] != w.WorldStates["free"])):
            return False
    return True


def moveAgent(agent, world):
    # Move a single agent if he can take his intended action
    x = agent.state[0]
    y = agent.state[1]
    if checkAgentMove(world, x, y, agent.action):
        if agent.action == ag.Actions["left"]:
            x = x - 1
        if agent.action == ag.Actions["right"]:
            x = x + 1
        if agent.action == ag.Actions["up"]:
            y = y - 1
        if agent.action == ag.Actions["down"]:
            y = y + 1
        if agent.action == ag.Actions["grab"]:
            print("grabbed it")
            agent.grasped = 1
            agent.reward = 1

        world.map[agent.state[0]][agent.state[1]] = w.WorldStates["free"]
        world.map[x][y] = w.WorldStates["agent"]
        agent.state = (x, y)


def moveBlock(agents, world):
    """Move to block according to the actions taken by the agents.

    For the action the agents want to do,
    check if it's possible for the agents and the block.
    If so,  first move the block and then let the agents move using moveAgent.
    """
    xblock = world.block[0]
    yblock = world.block[1]
    possibleMove = True

    # Set the state of the block on "free",
    # so the agents see that as movable space, this is reset later on
    world.map[world.block[0]][world.block[1]] = w.WorldStates["free"]

    for agent in agents:
        if not checkAgentMove(world, agent.state[0], agent.state[1], agent.action):
            possibleMove = False
            break

    if agents[0].action == ag.Actions["left"]:
        if xblock == 0 or world.map[xblock-1][yblock] == w.WorldStates["wall"]:
            possibleMove = False
        xblock = x - 1

    if agents[0].action == ag.Actions["right"]:
        if xblock == world.width - 1 or world.map[xblock+1][yblock] == w.WorldStates["wall"]:
            possibleMove = False
        xblock = x + 1

    if agents[0].action == ag.Actions["up"]:
        if yblock == 0 or world.map[xblock][yblock - 1] == w.WorldStates["wall"]:
            possibleMove = False
        yblock = y - 1

    if agents[0].action == ag.Actions["down"]:
        if yblock == world.height or world.map[xblock][yblock + 1] == w.WorldStates["wall"]:
            possibleMove = False
        yblock = y + 1

    if possibleMove:
        world.block = (xblock, yblock)

    for agent in agents:
        moveAgent(agent, world)

    world.map[world.block[0]][world.block[1]] = w.WorldStates["block"]


def updateWorld(agents, world):
    """Update everythings position in the world.

    If not all agents grabbed the block, move the single ones,
    else move the block if their actions correspond.
    """
    # If not all agents have grabbed the block,
    # move the single agents that have not grabbed. Else move the block.
    allGrasped = all(ag.grasped for ag in agents)
    sameAction = all(ag.action for ag in agents)
    if allGrasped and sameAction:
        moveBlock(agents, world)
    else:
        for agent in agents:
            if not agent.grasped:
                moveAgent(agent, world)
    return world

if __name__ == "__main__":

    # Some parameters
    numberOfAgents = 1
    tau = 0.99
    alpha = 0.5
    gamma = 0.5

    # World parameters
    height = 10
    width = 10
    goals = [(0, 4), (9, 0)]
    walls = [(2, 0), (2, 1), (2, 2), (2, 3), (2, 4)]
    block = (0, 3)

    # Starting positions of the agents,
    # should be the same amount as the number agents
    start = [(0, 0)]  # , (9, 9)]

    # Create the world and everything in it
    new_world = w.World(height, width, goals, walls, block, start)
    new_world.add_objects()
    new_world.print_map()

    # Create the agents
    agents = [ag.Agent(height*width, start[i], height, width)
              for i in range(numberOfAgents)]

    # Run a number of iterations
    # for x in range(50):
    while not any([new_world.block == g for g in goals]):
        # Choose an action for every agent
        [agent.chooseAction(tau) for agent in agents]
        # Perform the chosen actions (and obtain rewards)
        new_world = updateWorld(agents, new_world)
        # Update the Q-values of the agents
        [agent.updateQ(alpha, gamma) for agent in agents]

        tau = tau * 0.999
    # Print the updated map
    new_world.print_map()
