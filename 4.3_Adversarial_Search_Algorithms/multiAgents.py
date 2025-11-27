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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState


class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def getAction(self, gameState: GameState):
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
        bestIndices = [
            index for index in range(len(scores)) if scores[index] == bestScore
        ]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
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

        # Immediate win/lose states should dominate the evaluation
        if successorGameState.isWin():
            return float("inf")
        if successorGameState.isLose():
            return float("-inf")

        score = successorGameState.getScore()
        prevFoodCount = currentGameState.getNumFood()
        newFoodList = newFood.asList()
        newFoodCount = len(newFoodList)

        # Reward eating food and getting closer to the nearest pellet
        foodEaten = prevFoodCount - newFoodCount
        score += foodEaten * 120.0
        if newFoodList:
            nearestFoodDist = min(
                manhattanDistance(newPos, food) for food in newFoodList
            )
            score += 12.0 / max(1, nearestFoodDist)
            score -= 4.0 * newFoodCount

        # Encourage grabbing capsules soon
        prevCapsules = len(currentGameState.getCapsules())
        newCapsules = len(successorGameState.getCapsules())
        capsulesEaten = prevCapsules - newCapsules
        score += capsulesEaten * 150.0
        score -= 20.0 * newCapsules

        # Discourage stopping unless absolutely necessary
        if action == Directions.STOP:
            score -= 15.0

        # Factor ghost proximity and scared timers
        for ghostState, scaredTime in zip(newGhostStates, newScaredTimes):
            ghostPos = ghostState.getPosition()
            ghostDist = manhattanDistance(newPos, ghostPos)
            if scaredTime > 0:
                # Prefer chasing scared ghosts when close enough
                score += max(8 - ghostDist, 0) * 15.0
            else:
                if ghostDist == 0:
                    return float("-inf")
                if ghostDist <= 1:
                    score -= 250.0
                else:
                    score -= 20.0 / ghostDist

        return score


def scoreEvaluationFunction(currentGameState: GameState):
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

    def __init__(self, evalFn="scoreEvaluationFunction", depth="2"):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
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
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        numAgents = gameState.getNumAgents()

        def minimax(agentIndex: int, depth: int, state: GameState):
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state)

            actions = state.getLegalActions(agentIndex)
            if not actions:
                return self.evaluationFunction(state)

            nextAgent = (agentIndex + 1) % numAgents
            nextDepth = depth + 1 if nextAgent == 0 else depth

            if agentIndex == 0:
                value = float("-inf")
                for action in actions:
                    successor = state.generateSuccessor(agentIndex, action)
                    value = max(value, minimax(nextAgent, nextDepth, successor))
                return value

            value = float("inf")
            for action in actions:
                successor = state.generateSuccessor(agentIndex, action)
                value = min(value, minimax(nextAgent, nextDepth, successor))
            return value

        bestScore = float("-inf")
        bestAction = Directions.STOP
        for action in gameState.getLegalActions(0):
            successor = gameState.generateSuccessor(0, action)
            nextAgent = (0 + 1) % numAgents
            nextDepth = 1 if nextAgent == 0 else 0
            score = minimax(nextAgent, nextDepth, successor)
            if score > bestScore:
                bestScore = score
                bestAction = action

        return bestAction


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        numAgents = gameState.getNumAgents()

        def alphabeta(
            agentIndex: int, depth: int, state: GameState, alpha: float, beta: float
        ):
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state)

            actions = state.getLegalActions(agentIndex)
            if not actions:
                return self.evaluationFunction(state)

            nextAgent = (agentIndex + 1) % numAgents
            nextDepth = depth + 1 if nextAgent == 0 else depth

            if agentIndex == 0:
                value = float("-inf")
                for action in actions:
                    successor = state.generateSuccessor(agentIndex, action)
                    value = max(
                        value, alphabeta(nextAgent, nextDepth, successor, alpha, beta)
                    )
                    alpha = max(alpha, value)
                    if alpha > beta:
                        break
                return value

            value = float("inf")
            for action in actions:
                successor = state.generateSuccessor(agentIndex, action)
                value = min(
                    value, alphabeta(nextAgent, nextDepth, successor, alpha, beta)
                )
                beta = min(beta, value)
                if alpha > beta:
                    break
            return value

        alpha = float("-inf")
        beta = float("inf")
        bestScore = float("-inf")
        bestAction = Directions.STOP

        for action in gameState.getLegalActions(0):
            successor = gameState.generateSuccessor(0, action)
            nextAgent = (0 + 1) % numAgents
            nextDepth = 1 if nextAgent == 0 else 0
            score = alphabeta(nextAgent, nextDepth, successor, alpha, beta)
            if score > bestScore:
                bestScore = score
                bestAction = action
            alpha = max(alpha, score)

        return bestAction


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
    Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        numAgents = gameState.getNumAgents()

        def expectimax(agentIndex: int, depth: int, state: GameState):
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state)

            actions = state.getLegalActions(agentIndex)
            if not actions:
                return self.evaluationFunction(state)

            nextAgent = (agentIndex + 1) % numAgents
            nextDepth = depth + 1 if nextAgent == 0 else depth

            if agentIndex == 0:
                value = float("-inf")
                for action in actions:
                    successor = state.generateSuccessor(agentIndex, action)
                    value = max(value, expectimax(nextAgent, nextDepth, successor))
                return value

            probability = 1.0 / len(actions)
            value = 0.0
            for action in actions:
                successor = state.generateSuccessor(agentIndex, action)
                value += probability * expectimax(nextAgent, nextDepth, successor)
            return value

        bestScore = float("-inf")
        bestAction = Directions.STOP

        for action in gameState.getLegalActions(0):
            successor = gameState.generateSuccessor(0, action)
            nextAgent = (0 + 1) % numAgents
            nextDepth = 1 if nextAgent == 0 else 0
            score = expectimax(nextAgent, nextDepth, successor)
            if score > bestScore:
                bestScore = score
                bestAction = action

        return bestAction


def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: Blend the game score with weighted features: remaining food
    and capsules, distance to the closest pellets/power pellets, and ghost
    proximity (penalizing dangerous ghosts while rewarding reachable scared
    ghosts). Distances use Manhattan metrics with reciprocal shaping so nearer
    objectives dominate. Immediate win/lose states get +/-inf.
    """
    if currentGameState.isWin():
        return float("inf")
    if currentGameState.isLose():
        return float("-inf")

    position = currentGameState.getPacmanPosition()
    food = currentGameState.getFood().asList()
    capsules = currentGameState.getCapsules()
    ghostStates = currentGameState.getGhostStates()

    score = currentGameState.getScore()

    # Food heuristics: fewer pellets and shorter distance are better
    if food:
        foodDistances = [manhattanDistance(position, foodPos) for foodPos in food]
        closestFood = min(foodDistances)
        avgFoodDist = sum(foodDistances) / len(foodDistances)
        score += 14.0 / max(1, closestFood)
        score += 6.0 / max(1.0, avgFoodDist)
    score -= 6.0 * len(food)

    # Capsule pressure to encourage grabbing power pellets
    if capsules:
        capsuleDistances = [manhattanDistance(position, cap) for cap in capsules]
        closestCapsule = min(capsuleDistances)
        score += 18.0 / max(1, closestCapsule)
    score -= 20.0 * len(capsules)

    # Ghost handling
    for ghostState in ghostStates:
        ghostPos = ghostState.getPosition()
        dist = manhattanDistance(position, ghostPos)
        scaredTime = ghostState.scaredTimer
        if scaredTime > 0:
            # Prefer being near edible ghosts, especially when timer is high
            score += (scaredTime / max(1, dist)) * 12.0
        else:
            if dist == 0:
                return float("-inf")
            if dist <= 1:
                score -= 500.0
            else:
                score -= 16.0 / dist

    return score


# Abbreviation
better = betterEvaluationFunction
