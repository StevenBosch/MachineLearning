import world
import agent

""" Our actions are interpreted as the following integer values: """
Actions = {
    "stay": 0,
    "left": 1,
    "up": 2,
    "right": 3,
    "down": 4,
    "grab": 5
}

WorldStates = {
    "free": 0,
    "agent": 1,
    "wall": 2,
    "block": 3
}

def createAgents(numberOfAgents, start):
    agents = []    
    for i in range(numberOfAgents):
        agents.append(agent.Agent(height*width, start[i], height, width))
    return agents

def chooseActions(agents, tau):
    for agent in agents:
        agent.chooseAction(tau)

def updateWorld(agents, world):
    # Update the world according to the actions chosen by the agents
    for agent in agents:
        x = agent.state[0]
        y = agent.state[1]
        if agent.action == Actions["left"]:
            if not x == 0:
                if world.map[x-1][y] == WorldStates["free"]:
                    world.map[x][y] = WorldStates["free"]
                    x = x - 1
        if agent.action == Actions["right"]:
            if not x == world.width - 1:
                if world.map[x+1][y] == WorldStates["free"]:
                    world.map[x][y] = WorldStates["free"]
                    x = x + 1       
        if agent.action == Actions["up"]:
            if not y == 0:
                if world.map[x][y-1] == WorldStates["free"]:
                    world.map[x][y] = WorldStates["free"]
                    y = y - 1 
        if agent.action == Actions["down"]:
            if not x == world.height - 1:
                if world.map[x][y+1] == WorldStates["free"]:
                    world.map[x][y] = WorldStates["free"] 
                    y = y + 1                         
        if agent.action == Actions["grab"]:
            if nextToBlock(agent, world):
                agent.grasped = 1

        world.map[x][y] = WorldStates["agent"]
        agent.state = (x, y)
    pass
    
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
    new_world = world.World(height, width, goals, walls, block, start)
    new_world.add_objects()
    new_world.print_map()
    
    # Create the agents
    agents = createAgents(numberOfAgents, start)
    
    # Run the simulation a number of times
    for x in range(1000):
        chooseActions(agents, tau)
        updateWorld(agents, world)
        updateQValues(agents)

        tau = tau * 0.999