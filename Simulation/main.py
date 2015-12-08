import world
import agent

def createAgents(numberOfAgents):
    agents = []
    # Starting positions of the agents, should the same amount as agents
    start = [(0, 0), (9, 9)]
    for i in range(numberOfAgents):
        agents.append(agent.Agent(height*width, start[i], height, width))
    return agents

def chooseActions(agents, tau):
    for agent in agents:
        agent.chooseAction(tau)

def updateWorld(agents, world):
    # Update the world according to the actions chosen by the agents
    for agent in agents:
        if agent.action == 1:
            # The agent wants to go left
            if 
    
def updateQValues(agents):
    for agent in agents:
        agent.updateQ

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

    # Create world and everything in it
    new_world = world.World(height, width, goals, walls, block, start)
    new_world.add_objects()
    new_world.print_map()
    
    # Create the agents
    agents = createAgents(numberOfAgents)
    
    # Run the simulation a number of times
    for x in range(1000):
        chooseActions(agents, tau)
        updateWorld(agents, world)
        updateQValues(agents)

        tau = tau * 0.999