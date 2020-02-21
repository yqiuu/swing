import pytest
import numpy as np
from swing import ArtificialBeeColony, ParticleSwarm


def cost_funct(x):
    x = np.asarray(x)
    return np.sum(x*x)


def test_consistency(tmpdir):
    def run(optimizer, tmpfile):
        bounds = [(-2.2, 4.7)]*5
        #
        niter = 40
        rstate = np.random.RandomState(seed=20070831)
        op_0 = ArtificialBeeColony(cost_funct, bounds, rstate=rstate)
        op_0.swarm(niter=niter)
        memo_0 = dict(op_0.memo)
        for pos, cost in zip(memo_0['pos'], memo_0['cost']):
            assert np.isclose(cost, cost_funct(pos))
        #
        rstate = np.random.RandomState(seed=20070831)
        op_1 = ArtificialBeeColony(cost_funct, bounds, rstate=rstate)
        for i_iter in range(niter):
            op_1.swarm(niter=1)
        memo_1 = dict(op_1.memo)
        np.testing.assert_array_equal(memo_0['iter'], memo_1['iter'])
        np.testing.assert_array_equal(memo_0['ncall'], memo_1['ncall'])
        np.testing.assert_allclose(memo_0['pos'], memo_1['pos'])
        np.testing.assert_allclose(memo_0['cost'], memo_1['cost'])
        #
        niter_restart = 23
        rstate = np.random.RandomState(seed=20070831)
        op_2 = ArtificialBeeColony(cost_funct, bounds, rstate=rstate)
        op_2.swarm(niter=niter-niter_restart)
        op_2.save_checkpoint(tmpfile)
        op_2 = ArtificialBeeColony(cost_funct, bounds, restart_file=tmpfile)
        op_2.swarm(niter=niter_restart)
        memo_2 = dict(op_2.memo)
        np.testing.assert_array_equal(memo_0['iter'], memo_2['iter'])
        np.testing.assert_array_equal(memo_0['ncall'], memo_2['ncall'])
        np.testing.assert_allclose(memo_0['pos'], memo_2['pos'])
        np.testing.assert_allclose(memo_0['cost'], memo_2['cost'])

    tmpfile = tmpdir.mkdir('tmp').join('checkpoint')
    for op in ['ArtificialBeeColony', 'ParticleSwarm']:
        run(op, tmpfile)
