from .workspace import Workspace

import numpy as np


__all__ = ['SineCosine']


class SineCosine(Workspace):
    def __init__(self,
        func, bounds, nswarm=16, rstate=None, pool=None, vectorize=False, restart_file=None,
        factor_a=2., niter_max=100, initial_pos=None
    ):
        restart_keys = ['_pos', '_cost', "_niter_max"]
        super().__init__(
            func=func, bounds=bounds, nswarm=nswarm, rstate=rstate, pool=pool, vectorize=vectorize,
            restart_file=restart_file, restart_keys=restart_keys
        )
        self._factor_a = factor_a
        self._niter_max = niter_max
        self._initial_pos = initial_pos
        
    def _phase_init(self):
        if self._initial_pos is None:
            self._pos = self._init_new_pos(self._nswarm)
        else:
            self._pos = self._initial_pos
        self._cost = self._evaluate_multi(self._pos)

    def _phase_main(self):
        r1 = self._factor_a*(1 - self._i_iter/self._niter_max)

        next_pos = np.zeros([self._nswarm, self._ndim])
        cond = self._rstate.rand(self._nswarm) < 0.5
        # Sin
        n_update = np.count_nonzero(cond)
        r2 = 2*np.pi*self._rstate.rand(n_update, self._ndim)
        r3 = 2*self._rstate.rand(n_update, self._ndim)
        next_pos[cond] = self._pos[cond] \
            + r1*np.sin(r2)*np.abs(r3*self._pos_global_best - self._pos[cond])
        # Cos
        cond = ~cond
        n_update = np.count_nonzero(cond)
        r2 = 2*np.pi*self._rstate.rand(n_update, self._ndim)
        r3 = 2*self._rstate.rand(n_update, self._ndim)
        next_pos[cond] = self._pos[cond] \
            + r1*np.cos(r2)*np.abs(r3*self._pos_global_best - self._pos[cond])

        #
        next_pos = self._reflect_pos(next_pos)
        next_cost = self._evaluate_multi(next_pos)
        #
        self._pos = next_pos
        self._cost = next_cost