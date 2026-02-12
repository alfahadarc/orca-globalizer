import numpy as np
from orca_globalizer import GraphletGlobalizer, globalize_orca

def test_infers_4_node():
    X = np.zeros((3, 15))
    out = globalize_orca(X, orca_node=4)
    assert out.shape == (9,)

def test_infers_5_node():
    X = np.zeros((3, 73))
    out = globalize_orca(X, orca_node=5)
    assert out.shape == (30,)

def test_frequency_sums_to_one():
    X = np.ones((2, 15))
    out = globalize_orca(X, orca_node=4, method="frequency")
    assert np.isclose(out.sum(), 1.0)