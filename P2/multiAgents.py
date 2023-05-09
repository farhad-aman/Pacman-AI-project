# multiAgents.py
# --------------
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
import sys

from util import manhattanDistance
from game import Directions
import random, util

from game import Agent


class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        food_score = sys.maxsize
        for food in newFood.asList():
            distance = manhattanDistance(newPos, food)
            if distance < food_score:
                food_score = float(distance)

        ghost_distance_score = 0.0
        total_ghost_distances = 0.1
        ghosts = successorGameState.getGhostPositions()
        for ghost in ghosts:
            ghost_distance = manhattanDistance(ghost, newPos)
            total_ghost_distances += ghost_distance
            if ghost_distance <= 1:
                ghost_distance_score += -1

        scared_time_score = sum(newScaredTimes) / len(newScaredTimes)

        score = 0.0
        score += successorGameState.getScore()
        score += 2 * (1 / food_score)
        score += -2 * (1 / total_ghost_distances)
        score += 5 * ghost_distance_score
        score += scared_time_score
        return score


def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()


class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether the game state is a winning state

        gameState.isLose():
        Returns whether the game state is a losing state
        """
        action_scores = []
        num_agents = gameState.getNumAgents()

        def minimax(state, iter_count):
            if state.isWin() or state.isLose() or iter_count >= self.depth * num_agents:
                return self.evaluationFunction(state)
            agent_index = iter_count % num_agents
            if agent_index != 0:
                result = float('inf')
                for action in state.getLegalActions(agent_index):
                    if action != 'Stop':
                        new_state = state.generateSuccessor(agent_index, action)
                        result = min(result, minimax(new_state, iter_count + 1))
                return result
            else:
                result = float('-inf')
                for action in state.getLegalActions(agent_index):
                    if action != 'Stop':
                        new_state = state.generateSuccessor(agent_index, action)
                        result = max(result, minimax(new_state, iter_count + 1))
                        if iter_count == 0:
                            action_scores.append(result)
                return result

        minimax(gameState, 0)
        legal_actions = [a for a in gameState.getLegalActions(0) if a != 'Stop']
        best_action = legal_actions[action_scores.index(max(action_scores))]
        return best_action


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        score_list = []
        num_agents = gameState.getNumAgents()

        def alpha_beta(state, depth, alpha, beta):
            if state.isWin() or state.isLose() or depth >= self.depth * num_agents:
                return self.evaluationFunction(state)
            current_agent = depth % num_agents
            if current_agent != 0:
                result = sys.maxsize
                for action in state.getLegalActions(current_agent):
                    if action != "Stop":
                        new_state = state.generateSuccessor(current_agent, action)
                        alpha_beta_result = alpha_beta(new_state, depth + 1, alpha, beta)
                        if alpha_beta_result < result:
                            result = alpha_beta_result
                        if result < beta:
                            beta = result
                        if beta < alpha:
                            break
                return result
            else:
                result = -sys.maxsize
                for action in state.getLegalActions(current_agent):
                    if action != "Stop":
                        new_state = state.generateSuccessor(current_agent, action)
                        alpha_beta_result = alpha_beta(new_state, depth + 1, alpha, beta)
                        if alpha_beta_result > result:
                            result = alpha_beta_result
                        if result > alpha:
                            alpha = result
                        if depth == 0:
                            score_list.append(result)
                        if beta < alpha:
                            break
                return result

        alpha_beta(gameState, 0, -sys.maxsize, sys.maxsize)
        return [action for action in gameState.getLegalActions(0) if action != "Stop"][
            score_list.index(max(score_list))]


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        action_scores = []
        agents_count = gameState.getNumAgents()

        def expectiminimax(state, iter_count):
            if state.isWin() or state.isLose() or iter_count >= self.depth * agents_count:
                return self.evaluationFunction(state)

            iter_mod_agents = iter_count % agents_count
            if iter_mod_agents != 0:
                scores = []
                for action in state.getLegalActions(iter_mod_agents):
                    if action != "Stop":
                        new_state = state.generateSuccessor(iter_mod_agents, action)
                        score = expectiminimax(new_state, iter_count + 1)
                        scores.append(float(score))
                result = sum(scores) / len(scores)
                return result
            else:
                result = -sys.maxsize
                for action in state.getLegalActions(iter_mod_agents):
                    if action != "Stop":
                        new_state = state.generateSuccessor(iter_mod_agents, action)
                        score = expectiminimax(new_state, iter_count + 1)
                        if score > result:
                            result = score
                        if iter_count == 0:
                            action_scores.append(result)
                return result

        expectiminimax(gameState, 0)
        return [a for a in gameState.getLegalActions(0) if a != "Stop"][
            action_scores.index(max(action_scores))]


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    Don't forget to use pacmanPosition, foods, scaredTimers, ghostPositions!
    DESCRIPTION: <write something here so we know what you did>
    """

    pacmanPosition = currentGameState.getPacmanPosition()
    foods = currentGameState.getFood()
    ghostStates = currentGameState.getGhostStates()
    scaredTimers = [ghostState.scaredTimer for ghostState in ghostStates]
    ghostPositions = currentGameState.getGhostPositions()

    food_score = sys.maxsize
    for food in foods.asList():
        distance = manhattanDistance(pacmanPosition, food)
        if distance < food_score:
            food_score = float(distance)

    ghost_distance_score = 0.0
    total_ghost_distances = 0.1
    for ghost in ghostPositions:
        distance = util.manhattanDistance(pacmanPosition, ghost)
        total_ghost_distances += distance
        if distance <= 1:
            ghost_distance_score += -1

    scared_time_score = sum(scaredTimers) / len(scaredTimers)

    score = 0.0
    score += currentGameState.getScore()
    score += 2 * (1 / food_score)
    score += -2 * (1 / total_ghost_distances)
    score += 5 * ghost_distance_score
    score += scared_time_score

    return score


# Abbreviation
better = betterEvaluationFunction
