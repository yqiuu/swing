from .workspace import Workspace

import numpy as np


__all__ = ['ParticleSwarm']


class ParticleSwarm(Workspace):
    """
    Particle swarm optimiser.

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
    weight : float,
        Interial weight.
    acc_lbest : float
        Acceleration coefficient towards the local minimum.
    acc_gbest : float
        Acceleration coefficient towards the glaobl minimum.
    vel_max_frac : float
        Maximum velocity fraction with respect to the search ranges.
    """
    def __init__(self,
        func, bounds, nswarm=16, rstate=None, pool=None, vectorize=False, restart_file=None,
        weight=0.729, acc_lbest=1.49, acc_gbest=1.49, vel_max_frac=1., initial_pos=None
    ):
        restart_keys = [
            '_vel', '_pos', '_cost', '_pos_local_best', '_cost_local_best'
        ]
        super().__init__(
            func=func, bounds=bounds, nswarm=nswarm, rstate=rstate, pool=pool, vectorize=vectorize,
            restart_file=restart_file, restart_keys=restart_keys
        )
        self._weight = weight
        self._acc_lbest = acc_lbest
        self._acc_gbest = acc_gbest
        self._vel_max = vel_max_frac*self._dbounds
        if initial_pos is not None:
            initial_pos = initial_pos.copy()
        self._initial_pos = initial_pos


    def _init_new_vel(self):
        return np.zeros(self._ndim)


    def _next_vel(self, i_particle):
        pos = self._pos[i_particle]
        vel_max = self._vel_max
        acc_lbest = self._acc_lbest*self._rstate.rand(self._ndim)
        acc_gbest = self._acc_gbest*self._rstate.rand(self._ndim)
        vel = self._weight*self._vel[i_particle] \
            + acc_lbest*(self._pos_local_best[i_particle] - pos) \
            + acc_gbest*(self._pos_global_best - pos)
        # Check maximum velocity
        cond = vel > vel_max
        vel[cond] = vel_max[cond]
        cond = vel < -vel_max
        vel[cond] = -vel_max[cond]
        return vel


    def _check_bounds(self, vel, pos):
        vel = np.copy(vel)
        pos = np.copy(pos)
        for v_i, p_i in zip(vel, pos):
            cond = p_i < self._lbounds
            p_i[cond] = 2*self._lbounds[cond] - p_i[cond]
            v_i[cond] = -v_i[cond]
            cond = p_i > self._ubounds
            p_i[cond] = 2*self._ubounds[cond] - p_i[cond]
            v_i[cond] = -v_i[cond]
        return vel, pos


    def _update_local_best(self):
        for i_particle in range(self._nswarm):
            new_best = self._cost[i_particle]
            if new_best < self._cost_local_best[i_particle]:
                self._pos_local_best[i_particle] \
                    = np.copy(self._pos[i_particle])
                self._cost_local_best[i_particle] = new_best


    def _phase_init(self):
        self._vel = np.array(
            [self._init_new_vel() for i_swarm in range(self._nswarm)]
        )
        if self._initial_pos is None:
            self._pos = np.array(
                [self._init_new_pos() for i_swarm in range(self._nswarm)]
            )
        else:
            self._pos = self._initial_pos
        self._cost = self._evaluate_multi(self._pos)
        self._pos_local_best = np.copy(self._pos)
        self._cost_local_best = np.copy(self._cost)
        return {
            'init':{
                'vel': np.copy(self._vel),
                'pos': np.copy(self._pos),
                'cost': np.copy(self._cost)
            }
        }


    def _phase_main(self):
        next_vel = np.asarray(list(map(self._next_vel, range(self._nswarm))))
        next_pos = self._pos + next_vel
        next_vel, next_pos = self._check_bounds(next_vel, next_pos)
        next_cost = self._evaluate_multi(next_pos)
        self._vel = next_vel
        self._pos = next_pos
        self._cost = next_cost
        self._update_local_best()
        return {
            'main':{
                'vel': np.copy(self._vel),
                'pos': np.copy(self._pos),
                'cost': np.copy(self._cost)
            }
        }
