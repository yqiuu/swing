import pytest
import numpy as np
from swing import ArtificialBeeColony, ParticleSwarm, GreyWolf


def cost_func(x):
    x = np.asarray(x)
    return np.sum(x*x)


def compare_memo(memo_a, memo_b):
    np.testing.assert_array_equal(memo_a['iter'], memo_b['iter'])
    np.testing.assert_array_equal(memo_a['ncall'], memo_b['ncall'])
    np.testing.assert_allclose(memo_a['pos'], memo_b['pos'])
    np.testing.assert_allclose(memo_a['cost'], memo_b['cost'])


def test_consistency(tmpdir):
    def run(minimizer, tmpfile):
        bounds = [(-2.2, 4.7)]*5
        lb, ub = np.asarray(bounds).T
        # Run a fiducial model
        niter = 40
        rstate = np.random.RandomState(seed=20070831)
        op_0 = minimizer(cost_func, bounds, rstate=rstate)
        op_0.swarm(niter=niter)
        # Run each iteration manully
        rstate = np.random.RandomState(seed=20070831)
        op_1 = minimizer(cost_func, bounds, rstate=rstate)
        for i_iter in range(niter):
            info = op_1.swarm(niter=1)
            for data in info.values():
                for pos, cost in zip(data['pos'], data['cost']):
                    # Test consistency
                    assert(np.isclose(cost, cost_func(pos)))
                    # Test if all points are within the bounds
                    for p in pos:
                        assert(np.all(p >= lb) & np.all(p <= ub))
        compare_memo(op_0.memo, op_1.memo)
        # Test a restart run
        niter_restart = 23
        rstate = np.random.RandomState(seed=20070831)
        op_2 = minimizer(cost_func, bounds, rstate=rstate)
        op_2.swarm(niter=niter-niter_restart)
        op_2.save_checkpoint(tmpfile)
        op_2 = minimizer(cost_func, bounds, restart_file=tmpfile)
        op_2.swarm(niter=niter_restart)
        compare_memo(op_0.memo, op_2.memo)

    # Run tests for all optimizer
    tmpfile = tmpdir.mkdir('tmp').join('checkpoint')
    for target in [ArtificialBeeColony, ParticleSwarm, GreyWolf]:
        run(target, tmpfile)
