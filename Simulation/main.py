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
          (x == 0 or world.map[x-1][y] != w.WorldStates["free"])):
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
        if agent.action == ag.Actions["grab"] and agent.grasped == 0:
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
        xblock -= 1

    if agents[0].action == ag.Actions["right"]:
        if xblock == world.width - 1 or world.map[xblock+1][yblock] == w.WorldStates["wall"]:
            possibleMove = False
        xblock += 1

    if agents[0].action == ag.Actions["up"]:
        if yblock == 0 or world.map[xblock][yblock - 1] == w.WorldStates["wall"]:
            possibleMove = False
        yblock -= 1

    if agents[0].action == ag.Actions["down"]:
        if yblock == world.height - 1 or world.map[xblock][yblock + 1] == w.WorldStates["wall"]:
            possibleMove = False
        yblock += 1

    if possibleMove:
        world.block = (xblock, yblock)
        # Calculate
        if any(world.block == g for g in goals):
            for agent in agents:
                if agent.grasped:
                    agent.reward = 10
                    agent.updateQ(alpha, gamma)
        [moveAgent(agent, world) for agent in agents]

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
    height = 3
    width = 3
    goals = [(1, 1)]
    walls = []
    block = (0, 2)

    # Starting positions of the agents,
    # should be the same amount as the number agents
    start = [(0, 0)]  # , (9, 9)]

    # Create the agents
    agents = [ag.Agent(height*width, start[i], height, width)
              for i in range(numberOfAgents)]

    for epoch in range(100):
        print(epoch)
        # Set the agents to their starting positions
        for index, agent in enumerate(agents):
            agent.state = start[index]
            agent.grasped = 0

        # Create the world and everything in it
        new_world = w.World(height, width, goals, walls, block, start)
        new_world.add_objects()

        counter = 0;
        while not any([new_world.block == g for g in goals]):
            counter += 1
            # Choose an action for every agent
            [agent.chooseAction(tau, new_world) for agent in agents]
            # Perform the chosen actions (and obtain rewards)
            new_world = updateWorld(agents, new_world)
            # Update the Q-values of the agents
            [agent.updateQ(alpha, gamma) for agent in agents]

        tau = tau * 0.999

    agents[0].print_policy(new_world)
    new_world.print_map()

'''
Look at:
Where and when do the q-values have to get updated? Now on multiple occasions, which is incorrect.
Is the q-value calculation correct? (state vs nextState vs the place where updateQ is called, things are not in sync)
'''