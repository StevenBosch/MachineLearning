import world as w
import agent as ag


def createAgents(numberOfAgents, start):
    agents = []    
    for i in range(numberOfAgents):
        agents.append(ag.Agent(height*width, start[i], height, width))
    return agents

def chooseActions(agents, tau):
    for agent in agents:
        agent.chooseAction(tau)

def moveAgent(agent, world):
    reward = 0
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
            reward = 1

    world.map[x][y] = WorldStates["agent"]
    agent.state = (x, y)
    
    return reward

def moveBlock(agents, world):
    pass
        

def updateWorld(agents, world):
    # If not all agents have grabbed the block, move the single agents that have not grabbed. Else move the block.
    allGrasped = True
    sameAction = True
    action = agents[0].action
    for agent in agents:
        if agent.grasped == 0:
            agent.reward = moveAgent(agent, world)
            allGrasped = False
        elif agent.action != action:
            sameAction = False
    
    if allGrasped == True and sameAction == True:
        moveBlock(agents, world)
    
def updateQValues(agents):
    for agent in agents:
        agent.updateQ(reward, alpha, gamma)

if __name__ == "__main__":
    # Some parameters
    numberOfAgents = 2
    tau = 0.99
    
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
        updateQValues(agents)

        tau = tau * 0.999