# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util

from learningAgents import ValueEstimationAgent
import collections


class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """

    def __init__(self, mdp, discount=0.9, iterations=100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter()  # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        states = self.mdp.getStates()

        for iteration in range(self.iterations):
            tempvalues = util.Counter()
            for state in states:
                maxvalue = -999999
                actions = self.mdp.getPossibleActions(state)
                for action in actions:
                    transitionStatesProbs = self.mdp.getTransitionStatesAndProbs(state, action)
                    sumvalue = 0.0
                    for stateProb in transitionStatesProbs:
                        sumvalue += stateProb[1] * (
                                self.mdp.getReward(state, action, stateProb[0]) + self.discount * self.values[
                            stateProb[0]])
                    maxvalue = max(maxvalue, sumvalue)
                if maxvalue != -999999:
                    tempvalues[state] = maxvalue

            for state in states:
                self.values[state] = tempvalues[state]

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]

    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        transitionStatesProbs = self.mdp.getTransitionStatesAndProbs(state, action)
        value = 0.0
        for stateProb in transitionStatesProbs:
            value += stateProb[1] * (
                    self.mdp.getReward(state, action, stateProb[0]) + self.discount * self.values[stateProb[0]])
        return value
        # util.raiseNotDefined()

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        actions = self.mdp.getPossibleActions(state)
        maxaction = None
        maxvalueoveractions = -999999
        for action in actions:
            value = self.computeQValueFromValues(state, action)
            if value > maxvalueoveractions:
                maxvalueoveractions = value
                maxaction = action
        return maxaction
        # util.raiseNotDefined()

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)


class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """

    def __init__(self, mdp, discount=0.9, iterations=1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        states = self.mdp.getStates()
        for it in range(self.iterations):
            s = states[it % len(states)]
            if self.mdp.isTerminal(s):
                continue
            one_state = util.Counter()
            for a in self.mdp.getPossibleActions(s):
                one_state[a] = self.computeQValueFromValues(s, a)
            self.values[s] = max(one_state.values())


class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """

    def __init__(self, mdp, discount=0.9, iterations=100, theta=1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        states = self.mdp.getStates()
        pq = util.PriorityQueue()
        predec = {}

        for state in states:
            if self.mdp.isTerminal(state):
                continue
            q_values = {}
            for action in self.mdp.getPossibleActions(state):
                for next_state, prob in self.mdp.getTransitionStatesAndProbs(state, action):
                    if next_state not in predec:
                        predec[next_state] = set()
                    predec[next_state].add(state)
                q_values[action] = self.getQValue(state, action)
            diff = abs(self.values[state] - max(q_values.values()))
            pq.update(state, -diff)

        for it in range(self.iterations):
            if pq.isEmpty():
                break
            state = pq.pop()
            if self.mdp.isTerminal(state):
                continue
            update_values = {}
            for action in self.mdp.getPossibleActions(state):
                update_values[action] = self.getQValue(state, action)
            self.values[state] = max(update_values.values())
            for p in predec[state]:
                q_values = {}
                for action in self.mdp.getPossibleActions(p):
                    q_values[action] = self.getQValue(p, action)
                diff = abs(self.values[p] - max(q_values.values()))
                if diff > self.theta:
                    pq.update(p, -diff)
