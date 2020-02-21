from .workspace import Workspace

import numpy as np


__all__ = ['ParticleSwarm']


class ParticleSwarm(Workspace):
    def __init__(self,
        func, bounds, nswarm=32, rstate=None, pool=None, restart_file=None,
        weight=0.729, acc_lbest=1.49, acc_gbest=1.49, vel_max_frac=1.
    ):
        super().__init__(
            func=func, bounds=bounds, nswarm = nswarm, rstate=rstate, pool=pool,
            restart_file=restart_file,
            restart_keys=['_vel', '_pos', '_cost', '_pos_local_best', '_cost_local_best']
        )
        self._weight = weight
        self._acc_lbest = acc_lbest
        self._acc_gbest = acc_gbest
        self._vel_max = vel_max_frac*self._dbounds


    def _init_new_vel(self):
        return np.zeros(self._ndim)


    def _next_vel(self, i_particle):
        pos = self._pos[i_particle]
        vel_max = self._vel_max
        rstate = self._rstate
        #
        vel = self._weight*self._vel[i_particle] \
            + self._acc_lbest*rstate.rand(self._ndim)*(self._pos_local_best[i_particle] - pos) \
            + self._acc_gbest*rstate.rand(self._ndim)*(self._pos_global_best - pos)
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
                self._pos_local_best[i_particle] = np.copy(self._pos[i_particle])
                self._cost_local_best[i_particle] = new_best


    def _phase_init(self):
        self._vel = np.array([self._init_new_vel() for i_swarm in range(self._nswarm)])
        self._pos = np.array([self._init_new_pos() for i_swarm in range(self._nswarm)])
        self._cost = self._evaluate_multi(self._pos)
        self._pos_local_best = np.copy(self._pos)
        self._cost_local_best = np.copy(self._cost)
        return {'init': (np.copy(self._vel), np.copy(self._pos), np.copy(self._cost))}


    def _phase_main(self):
        next_vel = np.asarray(list(map(self._next_vel, range(self._nswarm))))
        next_pos = self._pos + next_vel
        next_vel, next_pos = self._check_bounds(next_vel, next_pos)
        next_cost = self._evaluate_multi(next_pos)
        self._vel = next_vel
        self._pos = next_pos
        self._cost = next_cost
        self._update_local_best()
        return {'main': (np.copy(self._vel), np.copy(self._pos), np.copy(self._cost))}