from .mdp import MDP
from .dist import DDist, uniform_dist
import numpy as np
import matplotlib.pyplot as plt

######################################################################
# Grid world
######################################################################

# A subclass of MDPs, defined by a simple 2D discrete grid map.  All
# have states of the form (i, j) indicating discrete indices into the
# map, and actions are 'n', 's', 'e', 'w'.


# Special characters in map
# '#' : wall, location is uninhabitable
# '*' : +50 reward, teleports to a random location
# 'G' : terminates
# '$' : +100 reward, terminates
# '@' : vortex, very hard to escape

class GridWorld(MDP):
    actions = ('n', 's', 'e', 'w')

    def __init__(self, charMap, discountFactor=0.9, ssp=False,
                     nom_prob = 0.5):
        self.cmap = charMap
        self.num_rows = len(charMap)
        self.num_cols = len(charMap[0])
        self.states = [(i, j) for i in range(self.num_cols) \
                       for j in range(self.num_rows)] + [(-1, -1)]
        self.q = None
        self.discount_factor = discountFactor
        self.start = uniform_dist(self.states)
        self.ssp = ssp
        self.nominal_trans_prob = nom_prob
        # graphics stuff
        self.ax = None
        self.colorBar = None

    def legal(self, s):
        (i, j) = s
        return (0 <= i < self.num_cols) and (0 <= j < self.num_rows) and \
               self.habitable(self.getChar(i, j))

    def terminal(self, s):
        (i, j) = s
        return i == -1 or self.getChar(i, j) in ('$', 'G')

    def habitable(self, c):
        return c != '#'

    def reward_fn(self, s, a):
        (i, j) = s
        if i == -1:
            # game over
            return 0
        elif self.ssp:
            # Treat this as a stochastic shortest paths problem, with
            # a reward of -1 for every step by default until termination
            return -1
        else:
            return self.charReward(self.getChar(i, j))

    def charReward(self, c):
        if c == '$':
            return 100
        elif c == '*':
            return 50
        else:
            return 0

    # i and j are coordinates where origin is at bottom left
    # i is in the traditional x direction, which is columns
    # j is in the traditional y direction, which is rows
    def getChar(self, i, j):
        # cmap is [row][column]
        # rows are flipped
        return self.cmap[self.num_rows - j - 1][i]

    # actions
    def aOffset(self, a):
        if a == 'n':
            return (0, 1)
        elif a == 's':
            return (0, -1)
        elif a == 'e':
            return (1, 0)
        else:
            return (-1, 0)

    def transition_model(self, s, a):
        (i, j) = s
        if self.terminal((i, j)) or not self.legal((i, j)):
            return DDist({(-1, -1): 1.0})

        if self.getChar(i, j) == '*':
            # Teleport!
            return uniform_dist([s for s in self.states if
                                 self.habitable(s)])

        if self.getChar(i, j) == '@':
            # Vortex!  Stay here with high probability
            stayProb = 0.99
            (ni, nj) = (i, j)
        else:
            stayProb = self.nominal_trans_prob  # Prob we move to the nominal
            #  next state
            (ai, aj) = self.aOffset(a)  # Offset due to action
            (ni, nj) = (i + ai, j + aj)  # Nominal next state
            if not self.legal((ni, nj)):
                # Destination not legal, stay centered here
                (ni, nj) = (i, j)

        neighbors = [nn for nn in \
                     [(ni + 1, nj), (ni - 1, nj), (ni, nj + 1), (ni, nj - 1)] \
                     if self.legal(nn)]
        result = {}
        result[(ni, nj)] = stayProb
        for nn in neighbors:
            result[nn] = (1 - stayProb) / len(neighbors)
        # print('s', s, 'a', a, 'nominal', (ni, nj), 'stayProb', stayProb)
        # print(result, sum(result.values()))
        # input('go')
        rdist = DDist(result)
        rdist.normalize()
        return rdist

    def rewardMap(self):
        if self.ax is None:
            plt.ion()
            plt.figure(facecolor="white")
            self.ax = plt.subplot()
        ima = np.array([[max(self.reward_fn((i, j), a) for a in self.actions) 
                         for i in range(self.num_cols)] \
                         for j in range(self.num_rows)])
        self.im = self.ax.imshow(np.flipud(ima),
                                 interpolation='none', cmap='viridis',
                                 extent=[-0.5, self.num_cols - 0.5,
                                         -0.5, self.num_rows - 0.5])

    def valueMap(self):
        if self.ax is None:
            plt.ion()
            plt.figure(facecolor="white")
            self.ax = plt.subplot()
        ima = np.array([[self.value((i, j)) \
                         for i in range(self.num_cols)] \
                        for j in range(self.num_rows)])
        self.im = self.ax.imshow(np.flipud(ima),
                                 interpolation='none', cmap='viridis',
                                 extent=[-0.5, self.num_cols - 0.5,
                                         -0.5, self.num_rows - 0.5])


    def policyMap(self, newAxes = False, clim_min = None, clim_max = None,
                  s = None):
        if newAxes or self.ax is None:
            plt.ion()
            plt.figure(facecolor="white")
            self.ax = plt.subplot()
        else:
            self.ax = plt.gca()
            self.ax.clear()

        c = {'n': 0, 's': 1, 'e': 2, 'w': 3}

        def xoff(a):
            if a == 'e':
                return .5
            elif a == 'w':
                return -.5
            else:
                return 0

        def yoff(a):
            if a == 'n':
                return .5
            elif a == 's':
                return -.5
            else:
                return 0

        X = range(self.num_cols)
        Y = range(self.num_rows)
        A = [[self.greedy((i, j)) \
              for i in range(self.num_cols)] \
             for j in range(self.num_rows)]
        # In yet another orientation.  Have to transpose
        U = [[xoff(A[j][i]) for i in range(self.num_cols)] \
             for j in range(self.num_rows)]
        V = [[yoff(A[j][i]) for i in range(self.num_cols)] \
             for j in range(self.num_rows)]
        self.ax.quiver(X, Y, U, V)
        ima = np.array([[self.value((i, j)) \
                         for i in range(self.num_cols)] \
                        for j in range(self.num_rows)])
        im = self.ax.imshow(np.flipud(ima),
                            interpolation='none', cmap='viridis',
                            extent=[-0.5, self.num_cols - 0.5,
                                    -0.5, self.num_rows - 0.5])
        # Used to make figuresn
        # im.set_clim(0, 300)
        if clim_min is not None:
            im.set_clim(clim_min, clim_max)

        if newAxes or self.colorBar is None:
            self.colorBar = plt.colorbar(im)
        else:
            self.colorBar.update_normal(im)
            #self.colorBar.on_mappable_changed(im)
       
       

# learn parameter is {False, 'Q', 'Dyna'}
def test_grid(grid, discountFactor = 0.9, ssp = False,
            learn = False, learningRate = 0.1,
            iters = 1000, eps_vi = 0.001, eps_ql = 0.1,
            interactive = False, learnVisUpdateRate = 500, newAxes = False,
            nom_prob = None,
            printFinal = False,
            clim_min = None, clim_max = None,
            learning_curve = False,
            policy_map = True):
    def interact(iter = 0, s = None, a = None, r = None):
        if not learn or (iter % learnVisUpdateRate == 0):
            if policy_map:
                m.policyMap(newAxes, s = s,
                        clim_min = clim_min, clim_max = clim_max)
                plt.title("Step:" + str(iter))
                if not newAxes:
                    plt.pause(0.1)
                    print('Step:', iter, 'Current state:', s)
                    input('go?')
                else:
                    plt.savefig('plot'+str(iter))
            if learning_curve:
                policy_evals.append((iter, m.policy_eval()))

    interactive_fn = interact if interactive else None
    policy_evals = []

    m = GridWorld(grid, discountFactor = discountFactor, ssp = ssp)
    if nom_prob is not None:
        m.nominal_trans_prob = nom_prob
    if learn == 'Q':    
        m.Q_learn(lr = learningRate, eps = eps_ql,
                  iters = iters, interactive_fn = interactive_fn)
    elif learn == 'Dyna':
        m.dyna_q(eps = eps_ql,
                 value_iteration_update_rate = 100,
                 iters = iters, interactive_fn = interactive_fn)
    else:
        m.value_iteration(eps = eps_vi, interactive_fn = interactive_fn,
                          max_iters = iters)

    if printFinal:
        for s in m.states:
            print(s, ':', m.greedy(s), ':', m.value(s))
    return policy_evals


smallGrid = ['....',
             '....',
             '..$.',
             '....']

medGrid1 = ['..........',
            '..........',
            '..........',
            '...$......',
            '..........',
            '..........',
            '..........',
            '.......$..',
            '..........',
            '..........']

medVortex = ['..........',
             '..........',
             '..........',
             '..........',
             '..........',
             '..........',
             '..........',
             '..@....$..',
             '..........',
             '..........']

medGrid2 = ['..........',
            '#####.###.',
            '..........',
            '..#####.##',
            '.....###..',
            '.......#..',
            '.....###..',
            '......#$..',
            '......#...',
            '..........']

medStar = ['..........',
           '........*.',
           '..........',
           '..........',
           '..........',
           '..........',
           '..........',
           '..........',
           '.$........',
           '..........']

medGoal100 = ['..........',
              '..........',
              '..........',
              '...$......',
              '..........',
              '..........',
              '..........',
              '..........',
              '..........',
              '..........']

medGoal = ['..........',
           '..........',
           '..........',
           '...G......',
           '..........',
           '..........',
           '..........',
           '..........',
           '..........',
           '..........']

def foo():
    fizz = GridWorld(smallGrid)
    fizz.rewardMap()

def test_easy_grid():
    return test_grid(medGrid1, interactive = True, iters = 10,
                         discountFactor = 0.9)

def test_grid_walls(eps = 0.00001, discountFactor = 0.9, iters = 50):
    return test_grid(medGrid2, eps_vi = eps, discountFactor = discountFactor,
                     interactive = True, iters = iters)

def test_grid_star(gamma = 0.9, iters = 50):
    return test_grid(medStar, discountFactor = gamma,
                     interactive = True, iters = iters)

def test_vortex(gamma = 0.9, iters = 50, nominal_trans_prob = None):
    return test_grid(medVortex, discountFactor = gamma,
                    interactive = True, iters = iters,
                   nom_prob = nominal_trans_prob)


# Learning examples

def test_small_grid_learn(ur = 1, iters = 200,
                          clim_min = None, clim_max = None):
    return test_grid(smallGrid, learn = True, iters = iters,
                         eps_ql = 0.1,
                        learnVisUpdateRate = ur, interactive = True,
                     clim_min = clim_min, clim_max = clim_max)

def test_easy_grid_learn():
    return test_grid(medGrid1, learn = True, iters = 50000,
                         eps_ql = 0.1,
                        learnVisUpdateRate = 200, interactive = True)

def test_grid_walls_learn():
    return test_grid(medGrid2, learn = True, iters = 5000000,
                         eps_ql = 0.1,
                        learnVisUpdateRate = 10000, interactive = True)

def testGridSSPLearn(iters = 100000, learn_vis_update_rate = 5000,
                         nom_prob = 0.8, eps = 0.1, interactive = False):
    return test_grid(medGoal, learn = True, ssp = True, iters = iters,
                         eps_ql = eps, learningRate = 0.5, 
                        learnVisUpdateRate = learn_vis_update_rate, newAxes = True, interactive = interactive, nom_prob = nom_prob)

def testGridGoalLearn(iters = 100000, learn_vis_update_rate = 5000,
                          nom_prob = 0.8, eps = 0.1, interactive = False):
    return test_grid(medGoal100, learn = True, iters = iters,
                         eps_ql = eps, learningRate = 0.5, 
                        learnVisUpdateRate = learn_vis_update_rate, newAxes = True, interactive = interactive, nom_prob = nom_prob)

def testSmallSSP(iters = 200, ur = 20):
    return test_grid(smallGrid, learn = True, iters = iters, ssp = True,
                        learnVisUpdateRate = ur, newAxes = True, interactive = True)

def Q_vs_Dyna(grid):
    def plotone(data, label):
        x = np.array([d[0] for d in data])
        means = np.array([d[1][0] for d in data])
        sterrs = np.array([d[1][1] for d in data])
        yl = means - sterrs
        yu = means + sterrs
        plt.fill_between(x, yl, yu, alpha=0.5, label = label)

    q_curve = test_grid(grid, learn = 'Q', iters = 30000, interactive = True,
                        learnVisUpdateRate = 1000, learning_curve = True,
                        policy_map=False)
    dyna_curve = test_grid(grid, learn = 'Dyna', iters = 30000, interactive = True,
                        learnVisUpdateRate = 1000, learning_curve = True,
                        policy_map=False)
    fig, ax = plt.subplots()
    plotone(q_curve, 'Q-learning')
    plotone(dyna_curve, 'Dyna-Q')
    ax.legend()
    ax.set_xlabel('Training steps')
    ax.set_ylabel('Average reward per step')
    plt.show()
    #return q_curve, dyna_curve



#test_easy_grid()
#test_grid_walls()
#test_small_grid_learn(ur = 5)
#test_grid_star(gamma = 1, iters = 20)
#test_grid_star(gamma = 1, iters = 20)
#test_small_grid_learn(ur = 10, iters = 200)
#test_grid_walls_learn()

