from .dist import uniform_dist, delta_dist, DDist, mixture_dist
from .util import argmax_with_val, argmax
import numpy as np

def rand_argmax(l, f):
    vals = np.array([f(x) for x in l])
    # Randomize in case of ties
    return l[np.random.choice(np.flatnonzero(vals == vals.max()))]

class MDP:
    # states: list or set of states
    # actions: list or set of actions
    # transitionModel: function from (state, action) into DDist over next state
    # rewardFn: function from (state, action) to real-valued reward
    # discountFactor: real, greater than 0, less than or equal to 1
    # startDist: optional instance of DDist, specifying initial state dist
    #    if it's unspecified, we'll use a uniform over states

    def __init__(self, states, actions, transition_model, reward_fn, 
                     discount_factor = 1.0, start_dist = None):
        self.states = states
        self.actions = actions
        self.transition_model = transition_model
        self.reward_fn = reward_fn
        self.discount_factor = discount_factor
        self.start = start_dist if start_dist else uniform_dist(states)
        self.q = None

    # Given a state, return True if the state should be considered to
    # be terminal.  You can think of a terminal state as generating an
    # infinite sequence of zero reward.
    def terminal(self, s):
        return False

    # Perform value iteration on this MDP.  Terminate when the
    # max-norm distance between two successive value function
    # estimates is less than eps.
    # interactiveFn is a function of zero arguments or None;  if it is
    # not None, it will be called once per iteration, for visuzalization

    # Sets self.q to be a dictionary mapping (s, a) pairs into Q values
    # This must be initialized before interactiveFn is called the first time.
    
    def value_iteration(self, eps = 0.0001, interactive_fn = None,
                        max_iters = 10000):
        mnd = float('inf')
        self.value_iteration_init()
        for it in range(max_iters):
            if interactive_fn: interactive_fn(it)
            new_q = self.value_iteration_update()
            mnd = max_norm_dist(self.q, new_q)
            if mnd  < eps:
                print('Value iteration finished.  Iterations:', it, 'eps', eps)
                return
            self.q = new_q
        print('Max norm dist is', mnd, 'after', max_iters, 'iters')

    def value_iteration_init(self, val = 0.0):
        self.q = dict([((s, a), val) for s in self.states for a in
                       self.actions])
    
    # Sparse because we probably won't have all the (s, a, s') triples
    def transition_model_init(self):
        self.t_counts = dict([((s, a), {}) for s in self.states for a in
                           self.actions])
        
    def transition_model_update(self, s, a, s_prime):
        if s_prime in self.t_counts[(s, a)]:
            self.t_counts[(s, a)][s_prime] += 1
        else:
            self.t_counts[(s, a)][s_prime] = 1

    def approx_transition_model(self, s, a):
        counts = self.t_counts[(s, a)]
        total = sum(counts.values())
        probs = dict([(s_prime, count / total) for (s_prime, count) in
                      counts.items()])
        return uniform_dist(self.states) if total == 0 else DDist(probs)

    def reward_model_init(self):
        self.r_counts = dict([((s, a), 0) for s in self.states for a in
                           self.actions])
        self.r_total = 0

    def reward_model_update(self, s, a, r):
        self.r_counts[(s, a)] += r        
        self.r_total += 1

    def value_iteration_update(self, use_true_model = True):
        new_q = {}
        t = self.transition_model if use_true_model else self.approx_transition_model
        for s in self.states:
            for a in self.actions:
                new_q[(s, a)] = self.reward_fn(s, a) + self.discount_factor * \
                        t(s, a).expectation(self.value)
        return new_q

    # Given a state, return the action that is greedy with reespect to the
    # current definition of the q function
    def greedy(self, s):
        return rand_argmax(self.actions, lambda a: self.q[(s, a)])

    # Given a state, return the value of that state, with respect to the 
    # current definition of the q function
    def value(self, s):
        return max(self.q[(s, a)] for a in self.actions)

    # Randomly choose a state from the initial state distribution
    def init_state(self):
        return self.start.draw()

    # Simulate a transition from state s, given action a.  Return
    # reward for (s,a) and new state, drawn from transition.

    # If a terminal state is encountered, get next state from initial state distribution
    def sim_transition(self, s, a):
        return (self.reward_fn(s, a),
                self.init_state() if self.terminal(s) else
                    self.transition_model(s, a).draw())

    # Q learning, handles terminal states
    def Q_learn(self, lr=.1, iters=1000, interactive_fn=None, eps=0.5, verbose=False):
        self.q = dict(
            [((s, a), 0.0) for s in self.states for a in self.actions])
        s = self.init_state()
        for i in range(iters):
            a = self.epsilon_greedy(s, eps)
            r, s_prime = self.sim_transition(s, a)
            future_val = 0 if self.terminal(s) else self.value(s_prime)
            self.q[(s, a)] = (1 - lr) * self.q[(s, a)] + \
                                 lr * (r + self.discount_factor * future_val)
            if interactive_fn:
                interactive_fn(i, s, a, r)
            s = s_prime
            if verbose:
                print(s, a, r, s_prime, self.q[(s, a)], self.terminal(s))

    def epsilon_greedy(self, s, eps=0.5):
        # Epsilon greedy action selection
        return mixture_dist(delta_dist(self.greedy(s)),
                            uniform_dist(self.actions), 1 - eps).draw()

    # Compute the q value of action a in state s with horizon h,
    # using expectimax
    def q_em(self, s, a, h):
        if h == 0:
            return 0
        else:
            return self.reward_fn(s, a) + \
                   self.discount_factor * \
                   sum([max(
                       [p * self.q_em(sp, ap, h - 1) for ap in self.actions]) \
                        for (sp, p) in self.transition_model(s, a).d.items()])

    def policy_em(self, s, h):
        return argmax(self.actions, lambda a: self.q_em(s, a, h))
    
    # Estimate the transition model and do value iteration sweeps
    def dyna_q(self, iters = 1000, eps = 0.5, value_iteration_update_rate = 100,
               interactive_fn = None):
        self.value_iteration_init()
        self.transition_model_init()
        self.reward_model_init()
        s = self.init_state()
        for i in range(iters):
            a = self.epsilon_greedy(s, eps)
            r, s_prime = self.sim_transition(s, a)
            self.transition_model_update(s, a, s_prime)
            self.reward_model_update(s, a, r)
            if i % value_iteration_update_rate == 0 and i > 0:
                self.q = self.value_iteration_update(use_true_model=False)
            if interactive_fn:
                interactive_fn(i, s, a, r)
            s = s_prime
    
    # Evaluate the policy encoded in the current Q values
    # Allow some exploration just to get out of stuck states
    def policy_eval(self, iters = 200, reps = 10, eps = 0.1, discount = 1.0):
        vals = []
        for j in range(reps):
            s = self.init_state()
            return_val = 0
            h = 0
            for i in range(iters):
                a = self.epsilon_greedy(s, eps)
                r, s_prime = self.sim_transition(s, a)
                s = s_prime
                return_val += discount ** h * r
                h = 0 if self.terminal(s) else h + 1
            # Should maybe be per episode if the problem is episodic
            vals.append(return_val / iters)
        mean = np.mean(vals)
        stderr = np.std(vals) / np.sqrt(reps)
        return mean, stderr

def max_norm_dist(q1, q2):
    u = list(set(q1.keys()) | set(q2.keys()))
    return argmax_with_val(u, lambda sa: abs(q1[sa] - q2[sa]))[1]

     
