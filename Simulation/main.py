import world as w
import agent as ag
import numpy as np

def createAgents(numberOfAgents, start):
    agents = []    
    for i in range(numberOfAgents):
        agents.append(ag.Agent(height*width, start[i], height, width))
    return agents

def chooseActions(agents, tau):
    for agent in agents:
        agent.chooseAction(tau)

def nextToBlock(agent, world):
        if np.absolute(agent.state[0]-world.block[0]) == 1:
                return True
        elif np.absolute(agent.state[1]-world.block[1]) == 1:
                return True
        else:
                return False

def moveAgent(agent, world):
    x = agent.state[0]
    y = agent.state[1]
    if agent.action == ag.Actions["left"]:
        if not x == 0:
            if world.map[x-1][y] == w.WorldStates["free"]:
                world.map[x][y] = w.WorldStates["free"]
                x = x - 1
    if agent.action == ag.Actions["right"]:
        if not x == world.width - 1:
            if world.map[x+1][y] == w.WorldStates["free"]:
                world.map[x][y] = w.WorldStates["free"]
                x = x + 1       
    if agent.action == ag.Actions["up"]:
        if not y == 0:
            if world.map[x][y-1] == w.WorldStates["free"]:
                world.map[x][y] = w.WorldStates["free"]
                y = y - 1 
    if agent.action == ag.Actions["down"]:
        if not x == world.height - 1:
            if world.map[x][y+1] == w.WorldStates["free"]:
                world.map[x][y] = w.WorldStates["free"] 
                y = y + 1                         
    if agent.action == ag.Actions["grab"]:
        if nextToBlock(agent, world):
            agent.grasped = 1
            agent.reward = 1

    world.map[x][y] = w.WorldStates["agent"]
    agent.state = (x, y)

def makeMoveAll(agents, world):
    pass

def moveBlock(agents, world):
    possibleMove = 1
    if agents[0].action == ag.Actions["left"]:
        for agent in agents:
            x = agent.state[0]
            y = agent.state[1]
            if x == 0 or world.map[x-1][y] == w.WorldStates["wall"]:
                possibleMove = 0
                break
    if agents[0].action == ag.Actions["right"]:
        for agent in agents:
            x = agent.state[0]
            y = agent.state[1]
            if x == world.width - 1 or world.map[x+1][y] == w.WorldStates["wall"]:
                possibleMove = 0
                break
    if agents[0].action == ag.Actions["up"]:
        for agent in agents:
            x = agent.state[0]
            y = agent.state[1]
            if y == 0 or world.map[x][y-1] == w.WorldStates["wall"]:
                possibleMove = 0
                break
    if agents[0].action == ag.Actions["down"]:
        for agent in agents:
            x = agent.state[0]
            y = agent.state[1]
            if y == world.height - 1 or world.map[x][y+1] == w.WorldStates["wall"]:
                possibleMove = 0
                break
    if possibleMove == 1:
        makeMoveAll(agents, world)

def updateWorld(agents, world):
    # If not all agents have grabbed the block, move the single agents that have not grabbed. Else move the block.
    allGrasped = True
    sameAction = True
    action = agents[0].action
    for agent in agents:
        agent.reward = 0
        if agent.grasped == 0:
            moveAgent(agent, world)
            allGrasped = False
        elif agent.action != action:
            sameAction = False
    
    if allGrasped == True and sameAction == True:
        moveBlock(agents, world)
    
def updateQValues(agents, alpha, gamma):
    for agent in agents:
        agent.updateQ(alpha, gamma)

if __name__ == "__main__":
    # Some parameters
    numberOfAgents = 2
    tau = 0.99
    alpha = 0.5
    gamma = 0.5
    
    # World parameters
    height = 10
    width = 10
    goals = [(0, 9), (9, 0)]
    walls = [(2, 0), (2, 1), (2, 2), (2, 3), (2, 4)]
    block = (5, 5)
    # Starting positions of the agents, should the same amount as agents
    start = [(0, 0), (9, 9)]

    # Create world and everything in it
    new_world = w.World(height, width, goals, walls, block, start)
    new_world.add_objects()
    new_world.print_map()
    
    # Create the agents
    agents = createAgents(numberOfAgents, start)
    
    # Run the simulation a number of times
    for x in range(1000):
        chooseActions(agents, tau)
        updateWorld(agents, new_world)
        updateQValues(agents, alpha, gamma)

        tau = tau * 0.999