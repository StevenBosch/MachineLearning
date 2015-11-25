import numpy
from enum import Enum

class Actions(Enum):
    stay = 0
    left = 1
    up = 2
    right = 3
    down = 4
    grab = 5
    release = 6

class Agent:
    def __init__(self, states, state, height, width):
        self.q = numpy.zeros((height, width, Actions.release.value + 1, 2))
        self.state = state
        self.grasped = 0
        self.action = 0

    def chooseAction(self, tau):
        sample = numpy.random.random_sample()
        probs = numpy.exp(self.q[self.state[0], self.state[1], :, self.grasped]/tau)/sum(numpy.exp(self.q[self.state[0], self.state[1], :, self.grasped]/tau))
        print(numpy.cumsum(probs))

        for action in Actions:
            if sample <= numpy.cumsum(probs)[action.value]:
                self.action = action.value

    def updateQ(self):
        pass

    def print_q(self):
        print(self.q)

    def print_trans(self):
        print(self.trans)
