"""Our agent module."""
import numpy as np
import world as w
from main import nextToBlock

""" Our actions are interpreted as the following integer values: """
Actions = {
    "stay": 0,
    "left": 1,
    "up": 2,
    "right": 3,
    "down": 4,
    "grab": 5
}

Actions2 = {
    0: "stay",
    1: "left",
    2: "up",
    3: "right",
    4: "down",
    5: "grab"
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

    def __init__(self, states, state, height, width):
        """ Initializes an agent with a starting state and parameters """
        self.q = np.zeros((width, height, len(Actions), 2))

        self.state = state
        self.nextState = (0, 0)
        self.action = 0
        self.grasped = 0
        self.reward = 0

    def chooseAction(self, tau, world):
        """ Chooses the best possible action, given a value of tau:
            Divide each number by the total sum, to achieve probabilities
            For each action a:
            P(a) = exp( Q(s, a), / tau) /
            SumAllA ( P(a) = exp( Q(s, a), / tau) )
            Probabilities = P(a) / SumAllA
        """
        if nextToBlock(self.state[0],self.state[1],world) and not self.grasped:
            self.action = Actions["grab"]
        else:
            sample = np.random.random_sample()
            single = np.exp(self.q[self.state[0], self.state[1], :,
                            self.grasped]/tau)
            total = sum(np.exp(self.q[self.state[0], self.state[1], :,
                               self.grasped]/tau))
            probs = single / total

            for action in Actions2:
                if sample <= np.cumsum(probs)[action]:
                    self.action = action
                    break

    def findMaxQ(self, state):
        """Find the maximum next q value, given the current state."""
        nextQ = -np.inf
        for action in Actions:
            qValue = self.q[state[0], state[1], Actions[action], self.grasped]
            nextQ = max(qValue, nextQ)
            return nextQ

    def updateQ(self, alpha, gamma):
        """ Updates a Q function:
            nextQ = Q[nextState][nextAction][obj];
            Q[state][action][obj] =
            curQ + ALPHA*(reward[obj] + GAMMA*nextQ - curQ);
        """
        state = self.state
        action = self.action
        nextState = self.nextState
        curQ = self.q[state[0], state[1], action, self.grasped]
        nextQ = self.findMaxQ(nextState)
        update = alpha * (self.reward + gamma * nextQ - curQ)
        self.q[state[0], state[1], action, self.grasped] = curQ + update
        self.reward = 0

    def print_q(self):
        """ Print the Q table """
        print(self.q)

    def print_policy(self, world):
        for g in range(2):
            print("Policy for grasped", g, ":")
            print("# "*(world.width+2))
            for x in range(world.width):
                print("#", end=" ")
                for y in range(world.height):
                    if np.argmax(self.q[x,y,:,g]) == 0.0: print("x", end=" ")
                    else: print(Actions3[np.argmax(self.q[x,y,:,g])], end = " ")

                print("#")
            print("# "*(world.width+2))
            print("\n")