from .workspace import Workspace

import numpy as np


__all__ = ['ArtificialBeeColony']


class ArtificialBeeColony(Workspace):
    """
    Artificial bee colony optimiser.

    Parameters
    ----------
    func : callable
        Cost function.
    bounds : list
        List of doublets (lower, upper) to specifiy the search ranges in each
        dimension.
    nswarm : int
        Number of particles.
    rstate : np.random.RandomState or None
        Random number generator. If None, ``np.random`` is used.
    pool : object
        Parallel executor that supports the ``map`` method.
    vectorize : bool
        If True, the target function can accept an array of variables and
        ``pool`` will be ignored.
    restart_file : str
        Path to the restart file.
    limit : int
        Number of failed searhes to start the scout phase.
    gbest_c : float
        Acceleration coefficient towards the glaobl minimum.
    """
    def __init__(self,
        func, bounds, nswarm=16, rstate=None, pool=None, vectorize=False,
        restart_file=None, limit=50, gbest_c=1.5, initial_pos=None,
        blob=False
    ):
        super().__init__(
            func=func,
            bounds=bounds,
            nswarm=nswarm,
            rstate=rstate,
            pool=pool,
            vectorize=vectorize,
            restart_file=restart_file,
            restart_keys=['_pos', '_cost', '_trail'],
            blob=blob
        )
        self._limit = limit
        self._gbest_c = gbest_c
        if initial_pos is not None:
            initial_pos = initial_pos.copy()
        self._initial_pos = initial_pos


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
        gbest_c = self._gbest_c*rstate.rand()
        new_pos_j = pos_ij \
            + (2*rstate.rand() - 1)*(pos[k_bee][j_dim] - pos_ij) \
            + gbest_c*(self._pos_global_best[j_dim] - pos_ij)
        lower = self._lbounds[j_dim]
        upper = self._ubounds[j_dim]
        if new_pos_j < lower:
            new_pos_j = 2*lower - new_pos_j
        elif new_pos_j > upper:
            new_pos_j = 2*upper - new_pos_j
        new_pos_j = max(new_pos_j, lower)
        new_pos_j = min(new_pos_j, upper)
        new_pos = np.copy(pos[i_bee])
        new_pos[j_dim] = new_pos_j
        return new_pos


    def _move(self, i_bee, new_pos, new_cost):
        if new_cost < self._cost[i_bee] or self._trail[i_bee] == -1:
            self._pos[i_bee] = new_pos
            self._cost[i_bee] = new_cost
            self._trail[i_bee] = 0
        else:
            self._trail[i_bee] += 1


    def _dance_area(self):
        fit = np.vectorize(
            lambda x: 1./(1. + x) if x > 0. else 1. + abs(x))(self._cost
        )
        fit /= np.sum(fit)
        return self._rstate.choice(self._nswarm, self._nswarm, True, fit)


    def _phase_init(self):
        if self._initial_pos is None:
            self._pos = self._init_new_pos(self._nswarm)
        else:
            self._pos = self._initial_pos
        self._cost, blob = self._evaluate_multi(self._pos)
        self._trail = np.zeros(self._nswarm, dtype='i4')
        return {
            'init':{
                'pos': np.copy(self._pos),
                'cost': np.copy(self._cost),
                'trail': np.copy(self._trail),
                "blob": blob
            }
        }


    def _phase_main(self):
        info = {}
        # Employer and scout bees phase
        queue = range(self._nswarm)
        new_pos = [None]*self._nswarm
        for i_bee in queue:
            if self._trail[i_bee] > self._limit:
                new_pos[i_bee] = self._init_new_pos()
                self._trail[i_bee] = -1
            else:
                new_pos[i_bee] = self._search_new_pos(i_bee)
        new_pos = np.asarray(new_pos)
        new_cost, blob = self._evaluate_multi(new_pos)
        for _ in map(self._move, queue, new_pos, new_cost): pass
        self._update_global_best()
        info['employer'] = {
            'pos': np.copy(new_pos),
            'cost': np.copy(new_cost),
            'trail': np.copy(self._trail),
            'blob': blob
        }
        # Onlooker bees phase
        queue = self._dance_area()
        new_pos = list(map(self._search_new_pos, queue))
        new_pos = np.asarray(new_pos)
        new_cost, blob = self._evaluate_multi(new_pos)
        for _ in map(self._move, queue, new_pos, new_cost): pass
        info['onlooker'] = {
            'pos': np.copy(new_pos),
            'cost': np.copy(new_cost),
            'trail': np.copy(self._trail),
            'blob': blob
        }
        return info
