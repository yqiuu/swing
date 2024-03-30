import pickle

import numpy as np
from scipy.stats.qmc import LatinHypercube
from tqdm import tqdm


__all__ = ['Workspace']


class Workspace:
    @property
    def pos_global_best(self):
        """Position of the global minimum"""
        return np.copy(self._pos_global_best)


    @property
    def cost_global_best(self):
        """Value of the the global minimum"""
        return self._cost_global_best


    @property
    def memo(self):
        """
        Number of calls, position and value of the global minimum at
        each iteration.
        """
        return {key:np.array(val) for key, val in self._memo.items()}


    def __init__(self,
        func, bounds, nswarm, rstate, pool, vectorize, restart_file, restart_keys
    ):
        self._func = func
        self._lbounds, self._ubounds = np.array(bounds).T
        self._dbounds = self._ubounds - self._lbounds
        self._ndim = len(bounds)
        self._nswarm = nswarm
        self._rstate = np.random.RandomState() if rstate is None else rstate
        self._lh_sampler = LatinHypercube(self._ndim, seed=self._rstate)
        self._vectorize = vectorize
        self._pool = pool
        #
        self._pos_global_best = np.full(self._ndim, np.nan)
        self._cost_global_best = np.inf
        self._pos = np.full((nswarm, self._ndim), np.nan)
        self._cost = np.full(nswarm, np.inf)
        #
        if restart_file is None:
            self._init_scheme = 'normal'
            self._i_iter = 0
            self._ncall = 0
            self._memo = {'iter': [], 'ncall': [], 'pos': [], 'cost': []}
        else:
            self._init_scheme = 'file'
            checkpoint = pickle.load(open(restart_file, 'rb'))
            for key, val in checkpoint.items():
                setattr(self, key, val)
            self._i_iter = self._memo['iter'][-1]
            self._ncall = self._memo['ncall'][-1]
        #
        self._restart_keys = list(restart_keys) \
            + ['_rstate', '_pos_global_best', '_cost_global_best', '_memo']


    def swarm(self, niter=100, progress_bar=True):
        """Run the optimizer.

        Parameters
        ----------
        niter : int
            Number of iterations to run.
        """
        def update_progress_bar(pbar):
            msg = "global minimum: {:.3e}".format(self.cost_global_best)
            pbar.set_description(msg)
            pbar.update()

        with tqdm(total=niter, disable=(not progress_bar)) as pbar:
            if self._init_scheme == 'normal':
                self._i_iter += 1
                niter -= 1
                info = self._phase_init()
                self._update_global_best()
                self._update_memo()
                update_progress_bar(pbar)

            for i_iter in range(niter):
                self._i_iter += 1
                info = self._phase_main()
                self._update_global_best()
                self._update_memo()
                update_progress_bar(pbar)

        self._init_scheme = 'restart'

        return info


    def save_checkpoint(self, fname):
        """Save a checkpoint.

        Parameters
        ----------
        fname : str
            Path of the checkpoint.
        """
        checkpoint = {}
        for key in self._restart_keys:
            checkpoint[key] = getattr(self, key)
        pickle.dump(checkpoint, open(fname, 'wb'))


    def _update_memo(self):
        self._memo['iter'].append(self._i_iter)
        self._memo['ncall'].append(self._ncall)
        self._memo['pos'].append(self._pos_global_best)
        self._memo['cost'].append(self._cost_global_best)


    def _evaluate(self, pos):
        self._ncall += 1
        return self._func(pos)


    def _evaluate_multi(self, pos):
        self._ncall += len(pos)
        if self._vectorize:
            return self._func(pos)
        elif self._pool is None:
            return np.asarray(list(map(self._func, pos)))
        else:
            return np.asarray(list(self._pool.map(self._func, pos)))


    def _update_global_best(self):
        idx_min = np.argmin(self._cost)
        cost_min = self._cost[idx_min]
        if cost_min < self._cost_global_best:
            self._pos_global_best = np.copy(self._pos[idx_min])
            self._cost_global_best = cost_min


    def _init_new_pos(self, num=1):
        if num == 1:
            samps = self._rstate.rand(self._ndim)
        else:
            samps = self._lh_sampler.random(num)
        return self._lbounds + self._dbounds*samps


    def _reflect_pos(self, pos):
        pos = np.copy(pos)
        for p_i in pos:
            cond = p_i < self._lbounds
            p_i[cond] = 2*self._lbounds[cond] - p_i[cond]
            cond = p_i > self._ubounds
            p_i[cond] = 2*self._ubounds[cond] - p_i[cond]
        return pos


    def _phase_init(self):
        return {'init': {'pos': None, 'cost': None}}


    def _phase_main(self):
        return {'main': {'pos': None, 'cost': None}}