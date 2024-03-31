from .workspace import Workspace

import numpy as np


__all__ = ['GreyWolf']


class GreyWolf(Workspace):
    def __init__(self,
        func, bounds, nswarm=16, rstate=None, pool=None, vectorize=False, restart_file=None,
        niter_max=100, initial_pos=None
    ):
        restart_keys = ['_pos', '_cost', "_niter_max"]
        super().__init__(
            func=func, bounds=bounds, nswarm=nswarm, rstate=rstate, pool=pool, vectorize=vectorize,
            restart_file=restart_file, restart_keys=restart_keys
        )
        self._niter_max = niter_max
        self._initial_pos = initial_pos

    def _phase_init(self):
        if self._initial_pos is None:
            self._pos = self._init_new_pos(self._nswarm)
        else:
            self._pos = self._initial_pos
        self._cost = self._evaluate_multi(self._pos)
        inds = np.argsort(self._cost)
        self._pos_alpha = np.copy(self._pos[inds[0]])
        self._cost_alpha = self._cost[inds[0]]
        self._pos_beta = np.copy(self._pos[inds[1]])
        self._cost_beta = self._cost[inds[1]]
        self._pos_delta = np.copy(self._pos[inds[2]])
        self._cost_delta = self._cost[inds[2]]

    def _phase_main(self):
        factor_a = 2*(1. - self._i_iter/self._niter_max)
        factor_a = max(0, factor_a)
        pos_1 = self._follow_leader(self._pos_alpha, factor_a)
        pos_2 = self._follow_leader(self._pos_beta, factor_a)
        pos_3 = self._follow_leader(self._pos_delta, factor_a)
        pos_new = (pos_1 + pos_2 + pos_3)/3.
        pos_new = self._reflect_pos(pos_new)

        self._pos = pos_new
        self._cost = self._evaluate_multi(self._pos)
        self._update_leader()

    def _update_leader(self):
        inds = np.argsort(self._cost)
        cost_sorted = self._cost[inds]
        pos_sorted = self._pos[inds]
        for p_i, c_i in zip(pos_sorted, cost_sorted):
            if c_i < self._cost_alpha:
                self._pos_alpha = p_i
                self._cost_alpha = c_i
                break
        for p_i, c_i in zip(pos_sorted, cost_sorted):
            if c_i >= self._cost_alpha and c_i < self._cost_beta:
                self._pos_beta = p_i
                self._cost_beta = c_i
                break
        for p_i, c_i in zip(pos_sorted, cost_sorted):
            if c_i >= self._cost_beta and c_i < self._cost_delta:
                self._pos_delta = p_i
                self._cost_delta = c_i
                break

    def _follow_leader(self, pos_leader, factor_a):
        factor_A = factor_a*(2*self._rstate.rand(self._nswarm, self._ndim) - 1)
        factor_C = 2*self._rstate.rand(self._nswarm, self._ndim)
        factor_D = np.abs(factor_C*pos_leader - self._pos)
        pos_new = pos_leader - factor_A*factor_D
        return pos_new