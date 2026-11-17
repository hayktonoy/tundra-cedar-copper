"""Synthetic checks for feature symmetry, interpolation and metric conventions."""
import json
import numpy as np
from features import raw_features, status_features, swap_index, interp_ball, degrade
from run import decode, score, LOCAL, save, predict

rng = np.random.default_rng(827)
p = np.full((501, 2, 32, 2), np.nan, 'f4')
p[:, :, :11] = rng.normal(0, 10, (1, 2, 11, 2)) + np.arange(501)[:, None, None, None]*.003
b = rng.normal(0, .2, (501, 2)).astype('f4')
b[100:180] = np.nan
x, names = raw_features(p, b)
xx, nn = raw_features(p[:, ::-1], b)
assert nn == names
np.testing.assert_allclose(xx, x[:, swap_index(names)], equal_nan=True)
np.testing.assert_allclose(status_features(x, names), status_features(xx, names), equal_nan=True)
assert not np.isinf(x).any()
np.testing.assert_allclose(interp_ball(np.array([[0., 1], [np.nan, np.nan], [2, 3]], 'f4')), [[0, 1], [1, 2], [2, 3]])
for cfg in [(1, .5, 1), (7, .65, 5)]:
    np.testing.assert_array_equal(decode(np.tile([.99, .5], (101, 1)), cfg), 2)
    np.testing.assert_array_equal(decode(np.tile([.01, .99], (101, 1)), cfg), 0)
    np.testing.assert_array_equal(decode(np.tile([.01, .01], (101, 1)), cfg), 1)
y = np.array([0]*40+[2]*20+[1]*40, 'i1')
assert score(y, y)['spell'] == 1
assert score(y, y)['macro'] == 1
sensor = dict(mask=np.ones(501, bool), counts=np.full((501, 2), 10, 'i1'), residual=np.zeros((501, 2), 'f4'))
pp, bb = degrade(p, b, sensor, 10)
assert np.max(np.isfinite(pp).all(3).sum(2)) == 10
save(LOCAL/'checks.json', dict(passed=['team feature swap', 'status invariance', 'finite values', 'gap interpolation', 'constant decoder', 'official metric perfect identity', 'player degradation'], features=len(names)))
print('7 synthetic checks passed;', len(names), 'features')
if (LOCAL/'models/inner_J03WMX/done.json').exists():
    import lightgbm as lgb
    models = [lgb.Booster(model_file=str(LOCAL/'models/inner_J03WMX'/f'{h}.txt')) for h in ('dead', 'owner')]
    p1 = predict(models, x, names); p2 = predict(models, xx, names)
    np.testing.assert_allclose(p1[:, 0], p2[:, 0], atol=1e-7)
    np.testing.assert_allclose(p1[:, 1]+p2[:, 1], 1, atol=1e-7)
    assert np.isfinite(p1).all() and p1.min()>=0 and p1.max()<=1
    save(LOCAL/'probability_checks.json', dict(dead_swap_invariant=True, home_swap_complementary=True, finite_unit_interval=True))
    print('Actual fitted-head probability symmetry checks passed')
