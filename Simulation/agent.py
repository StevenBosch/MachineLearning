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

    def __init__(self, state, rows, columns):
        """ Initializes an agent with a starting state and parameters """
        self.q = np.zeros((rows, columns, len(Actions), 2))

        self.prevState = (0, 0)
        self.state = state
        self.action = 0
        self.prevGrasped = 0
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
        if nextToBlock(self.state[0], self.state[1], world) and not self.grasped:
            self.action = Actions["grab"]
        else:
            sample = np.random.random_sample()
            single = np.exp(self.q[self.state[0], self.state[1], :,
                            self.grasped]/tau)
            if any(single == float('inf')):
                print("tau ", tau)
                print("q values ", self.q[self.state[0], self.state[1], :,
                                          self.grasped])
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

        We calculate the new q-value for the PREVIOUS state
        nextQ = Q[nextState][nextAction][obj];
        Q[state][action][obj] =
        curQ + ALPHA*(reward[obj] + GAMMA*nextQ - curQ);
        """
        state = self.prevState
        action = self.action
        nextState = self.state
        curQ = self.q[state[0], state[1], action, self.prevGrasped]
        nextQ = self.findMaxQ(nextState)
        update = alpha * (self.reward + gamma * nextQ - curQ)
        self.q[state[0], state[1], action, self.prevGrasped] = curQ + update
        self.reward = 0

    def print_q(self):
        """ Print the Q table """
        print(self.q)

    def print_policy(self, world):
        for g in range(2):
            print("Policy for grasped", g, ":")
            print("# "*(world.columns+2))
            for y in range(world.rows):
                print("#", end=" ")
                for x in range(world.columns):
                    if np.argmax(self.q[y, x, :, g]) == 0.0:
                        print("x", end=" ")
                    else:
                        print(Actions3[np.argmax(self.q[y, x, :, g])], end = " ")

                print("#")
            print("# "*(world.columns+2))
            print("\n")