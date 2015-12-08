import numpy as np

Actions = {
    "stay": 0,
    "left": 1,
    "up": 2,
    "right": 3,
    "down": 4,
    "grab": 5,
    "release": 6,
}


class Agent:
    def __init__(self, states, state, height, width):
        self.q = np.zeros((height, width, Actions.release.value + 1, 2))
        self.state = state
        self.grasped = 0
        self.action = 0

    def chooseAction(self, tau):
        sample = np.random.random_sample()
        # Divide each number by the total sum, to achieve probabilities
        # For each action a:
        # P(a) = exp( Q(s, a), / tau) /
        # SumAllA ( P(a) = exp( Q(s, a), / tau) )
        single = np.exp(self.q[self.state[0], self.state[1], :, self.grasped]/tau)
        sum = sum(np.exp(self.q[self.state[0], self.state[1], :, self.grasped]/tau))
        probs = single / sum
        print(np.cumsum(probs))

        for action in Actions:
            if sample <= np.cumsum(probs)[action.value]:
                self.action = action.value
    def updateQ(self, state, action):

        pass

    def print_q(self):
        print(self.q)

    def print_trans(self):
        print(self.trans)
