from .workspace import Workspace

import numpy as np


__all__ = ['ArtificialBeeColony']


class ArtificialBeeColony(Workspace):
    def __init__(self,
        func, bounds, nswarm=32, rstate=None, pool=None, restart_file=None, limit=0, gbest_c=1.5
    ):
        super().__init__(
            func=func, bounds=bounds, nswarm=nswarm, rstate=rstate, pool=pool,
            restart_file=restart_file, restart_keys=['_pos', '_cost', '_trail']
        )
        if limit == 0:
            self._limit = int(.6*nswarm*self._ndim)
        else:
            self._limit = limit
        self._gbest_c = gbest_c

    
    def _search_new_pos(self, i_bee):
        rstate = self._rstate
        pos = self._pos
        # Select a dimension
        j_dim = rstate.randint(self._ndim)
        # Select a bee
        while True:
            k_bee = rstate.randint(self._nswarm)
            if k_bee != i_bee:
                break
        # Generate a new position
        pos_ij = pos[i_bee][j_dim]
        new_pos_j = pos_ij + (2*rstate.rand() - 1)*(pos[k_bee][j_dim] - pos_ij) \
            + self._gbest_c*rstate.rand()*(self._pos_global_best[j_dim] - pos_ij)
        lower = self._lbounds[j_dim]
        upper = self._ubounds[j_dim]
        if new_pos_j < lower:
            new_pos_j = lower
        elif new_pos_j > upper:
            new_pos_j = upper
        new_pos = np.copy(pos[i_bee]) 
        new_pos[j_dim] = new_pos_j
        return new_pos


    def _move(self, i_bee, new_pos, new_cost):
        if new_cost < self._cost[i_bee]:
            self._pos[i_bee] = new_pos
            self._cost[i_bee] = new_cost
            self._trail[i_bee] = 0
        else:
            self._trail[i_bee] += 1


    def _dance_area(self):
        fit = np.vectorize(lambda x: 1./(1. + x) if x > 0. else 1. + abs(x))(self._cost)
        fit /= np.sum(fit)
        return self._rstate.choice(self._nswarm, self._nswarm, True, fit)


    def _phase_init(self):
        self._pos = np.asarray([self._init_new_pos() for i_bee in range(self._nswarm)])
        self._cost = self._evaluate_multi(self._pos)
        self._trail = np.zeros(self._nswarm, dtype='i4')
        return {'init': (np.copy(self._pos), np.copy(self._cost), np.copy(self._trail))}


    def _phase_main(self):
        info = {}
        # Employer and scout bees phase
        queue = range(self._nswarm)
        new_pos = [None]*self._nswarm
        for i_bee in queue:
            if self._trail[i_bee] > self._limit:
                new_pos[i_bee] = self._init_new_pos()
                self._trail[i_bee] = 0
            else:
                new_pos[i_bee] = self._search_new_pos(i_bee)
        new_cost = self._evaluate_multi(new_pos)
        for _ in map(self._move, queue, new_pos, new_cost): pass
        self._update_global_best()
        info['employer'] = (np.copy(self._pos), np.copy(self._cost), np.copy(self._trail))
        # Onlooker bees phase
        queue = self._dance_area()
        new_pos = list(map(self._search_new_pos, queue))
        new_cost = self._evaluate_multi(new_pos)
        for _ in map(self._move, queue, new_pos, new_cost): pass
        info['onlooker'] = (np.copy(self._pos), np.copy(self._cost), np.copy(self._trail))
        return info
