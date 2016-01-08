"""Our self module."""
import numpy as np
import world as w

""" Our actions are interpreted as the following integer values: """
Actions = {
    "stay": 0,
    "left": 1,
    "up": 2,
    "right": 3,
    "down": 4,
    "grab": 5
}

Actions3 = {
    0: "s",
    1: "l",
    2: "u",
    3: "r",
    4: "d",
    5: "g"
}


class Agent:
    """Our agent class"""

    def __init__(self, startStates, rows, columns, nAgents):
        """ Initializes an agent with a starting state and parameters """
        # Q can be indexed by [x, y for every agent, action for every agent,
        # grasped for every agent], i.e.
        # q[0,0,1,1,4,5,0,0] for 2 agents
        self.nAgents = nAgents
        self.q = np.zeros((
            [rows, columns] * nAgents +
            [2] * nAgents +
            [len(Actions) ** nAgents]
        ))

        self.prevState = [0, 0] * nAgents
        # State contains the x,y coodinates of all agents
        self.state = []
        for i in range(self.nAgents):
            self.state += list(startStates[i])

        # Action contains the action for every agent
        self.action = 0
        self.prevGrasped = [0] * nAgents
        self.grasped = [0] * nAgents
        self.reward = 0
        self.startState = self.state
        
    def reset(self):
        self.state = self.startState
        self.grasped = [0] * self.nAgents

    def valueToActionList(self, value):
        actions = []
        for i in range(self.nAgents):
            actions.append(
                value // (len(Actions) ** (self.nAgents - i - 1)) % len(Actions)
            )
        return actions

    def chooseAction(self, tau, world):
        """ Chooses the best possible action, given a value of tau:
            Divide each number by the total sum, to achieve probabilities
            For each action a:
            P(a) = exp( Q(s, a), / tau) /
            SumAllA ( P(a) = exp( Q(s, a), / tau) )
            Probabilities = P(a) / SumAllA
        """
        sample = np.random.random_sample()
        index = tuple(self.state + self.grasped)

        single = np.exp(self.q[index]/tau)

        # Print for debugging inf values
        if any(single == float('inf')):
            print("tau ", tau)
            print("q values ", self.q[index])

        total = sum(single)
        probs = single / total

        cumSumProbs = np.cumsum(probs)
        for action in range(len(cumSumProbs)):
            if sample <= cumSumProbs[action]:
                self.action = action
                break

    def chooseGreedyAction(self, epsilon, world):
        """ Chooses the next action, greedy style

        With a chance epsilon, choose the best action, else
        choose a random action
        """
        sample = np.random.random_sample()
        index = tuple(self.state + self.grasped)
        if sample < epsilon:
            self.action = np.argmax(self.q[index])
        else:
            self.action = np.random.randint(len(Actions)**self.nAgents)

    def updateQ(self, alpha, gamma):
        """ Updates a Q function:

        We calculate the new q-value for the PREVIOUS state
        nextQ = Q[nextState][nextAction][obj];
        Q[state][action][obj] =
        curQ + ALPHA*(reward[obj] + GAMMA*nextQ - curQ);
        """
        state = self.prevState
        action = self.action
        nextState = self.state

        currentIndex = tuple(state + self.grasped + [self.action])
        curQ = self.q[currentIndex]
        nextIndex = tuple(state + self.grasped)
        nextQ = np.max(self.q[nextIndex])

        update = alpha * (self.reward + gamma * nextQ - curQ)
        self.q[currentIndex] = curQ + update
        self.reward = 0

    def print_q(self):
        """ Print the Q table """
        print(self.q)

    def print_policy(self, world):
        for g in range(2):
            print("Policy for grasped", g, ":")
            print("# "*(world.columns+2))
            for y in range(world.rows):
                print("#", end=' ')
                for x in range(world.columns):
                    if np.argmax(self.q[y, x, :, g]) == 0.0:
                        print("x", end=" ")
                    else:
                        print(Actions3[np.argmax(self.q[y, x, :, g])], end=" ")

                print("#")
            print("# "*(world.columns+2))
            print("\n")

    def moveAgent(self, world):
        # Move a single agent if he can take his intended action
        actions = self.valueToActionList(self.action)
        for agent in range(self.nAgents):
            y = self.state[2*agent]
            x = self.state[2*agent+1]
            self.prevGrasped[agent] = self.grasped[agent]
            #print("y: ", y, " x: ", x, "action: ", actions[agent])

            if world.checkMove(y, x, actions[agent]):
                if actions[agent] == Actions["left"]:
                    x = x - 1
                if actions[agent] == Actions["right"]:
                    x = x + 1
                if actions[agent] == Actions["up"]:
                    y = y - 1
                if actions[agent] == Actions["down"]:
                    y = y + 1
                if actions[agent] == Actions["grab"] and self.grasped == 0:
                    self.grasped = 1
                    self.reward += 1

                world.map[self.state[2*agent]][self.state[2*agent+1]] = w.WorldStates["free"]
                world.map[y][x] = w.WorldStates["agent"]
                self.state[2*agent] = y
                self.state[2*agent+1] = x
