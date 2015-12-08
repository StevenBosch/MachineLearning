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

def nextToBlock(x, y, world):
    if np.absolute(x-world.block[0]) + np.absolute(y-world.block[1]) == 1:
        return True
    return False

def checkAgentMove(world, x, y, direction):
    # Check to see if an agent can make a desired move
    if direction == ag.Actions["left"]:
        if x == 0:
            return False
        elif world.map[x-1][y] == w.WorldStates["free"]:
            return True
    if direction == ag.Actions["right"]:
        if x == world.width - 1:
            return False
        elif world.map[x+1][y] == w.WorldStates["free"]:
            return True
    if direction == ag.Actions["up"]:
        if y == 0:
            return False
        elif world.map[x][y-1] == w.WorldStates["free"]:
            return True
    if direction == ag.Actions["down"]:
        if y == world.height - 1:
            return False
        elif world.map[x][y+1] == w.WorldStates["free"]:
            return True
    if direction == ag.Actions["grab"]:
        if nextToBlock(x, y, world):
            return True
    return False

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
            agent.grasped = 1
            agent.reward = 1
    
        world.map[agent.state[0]][agent.state[1]] = w.WorldStates["free"]
        world.map[x][y] = w.WorldStates["agent"]
        agent.state = (x, y)

def moveBlock(agents, world):
    # For the action the agents want to do, check if it's possible for the agents and the block.
    # If so,  first move the block and then let the agents move using moveAgent.
    xblock = world.block[0]
    yblock = world.block[1]
    possibleMove = True
    
    # Set the state of the block on "free" so the agents see that as movable space, this is reset later on
    world.map[world.block[0]][world.block[1]] = WorldStates["free"]
    
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
    
    world.map[world.block[0]][world.block[1]] = WorldStates["block"]

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
    
    # Starting positions of the agents, should be the same amount as the number agents
    start = [(0, 0), (9, 9)]

    # Create the world and everything in it
    new_world = w.World(height, width, goals, walls, block, start)
    new_world.add_objects()
    new_world.print_map()
    
    # Create the agents
    agents = createAgents(numberOfAgents, start)
    
    # Run a number of iterations
    for x in range(1000):
        chooseActions(agents, tau)
        updateWorld(agents, new_world)
        updateQValues(agents, alpha, gamma)

        tau = tau * 0.999