"""Our agent module."""
import numpy as np

""" Our actions are interpreted as the following integer values: """
Actions = {
    "stay": 0,
    "left": 1,
    "up": 2,
    "right": 3,
    "down": 4,
    "grab": 5
}


class Agent:
    """Our agent class"""

    def __init__(self, states, state, height, width):
        """ Initializes an agent with a starting state and parameters """
        self.q = np.zeros((height, width, len(Actions), 2))

        self.state = state
        self.nextState = (0, 0)
        self.action = 0
        self.newState = 0
        self.grasped = 0
        self.reward = 0

    def chooseAction(self, tau):
        """ Chooses the best possible action, given a value of tau:
            Divide each number by the total sum, to achieve probabilities
            For each action a:
            P(a) = exp( Q(s, a), / tau) /
            SumAllA ( P(a) = exp( Q(s, a), / tau) )
            Probabilities = P(a) / SumAllA
        """
        sample = np.random.random_sample()
        single = np.exp(self.q[self.state[0], self.state[1], :,
                        self.grasped]/tau)
        total = sum(np.exp(self.q[self.state[0], self.state[1], :,
                           self.grasped]/tau))
        probs = single / total
        # print(np.cumsum(probs))

        for action in Actions:
            if sample <= np.cumsum(probs)[Actions[action]]:
                self.action = Actions[action]

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

    def print_q(self):
        """ Print the Q table """
        print(self.q)
