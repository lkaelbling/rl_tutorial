import grid_worlds as gw
import numpy as np
import matplotlib.pyplot as plt


# Things to try for value iteration:
# 1. Make new grid layouts
# 2. Change the discount factor 
# 3. Change the nominal transition probability
gw.test_grid(gw.medVortex, interactive = True, nom_prob = 0.8, discountFactor=0.9)

# By default, draws after 10K steps
#gw.test_grid_walls_learn()

'''
gw.test_grid(gw.medGrid2, 
             learn = 'Q', 
             interactive = True, 
             iters = 50000,
             learnVisUpdateRate=1000,
             learning_curve=True,
             policy_map = False)
'''

#gw.Q_vs_Dyna(gw.medGrid2)
pass


