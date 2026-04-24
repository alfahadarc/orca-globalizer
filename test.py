import numpy as np

from orca_globalizer import GraphletGlobalizer, globalize_orca


def test_4_node_orbits() -> None:
    x = np.zeros((3, 15))
    out = globalize_orca(x, orca_node=4)
    assert out.shape == (9,)
    assert np.all(out == 0)


def test_5_node_orbits() -> None:
    x = np.zeros((2, 73))
    out = globalize_orca(x, orca_node=5)
    assert out.shape == (30,)
    assert np.all(out == 0)


def test_frequency_mode_sums_to_one() -> None:
    x = np.ones((2, 15))
    out = globalize_orca(x, orca_node=4, method="frequency")
    assert np.isclose(out.sum(), 1.0)


def test_class_api() -> None:
    x = np.ones((1, 15))
    globalizer = GraphletGlobalizer()
    out, labels = globalizer.transform(x, return_labels=True)
    assert len(out) == 9
    assert labels == [f"G{i}" for i in range(9)]


if __name__ == "__main__":
    test_4_node_orbits()
    test_5_node_orbits()
    test_frequency_mode_sums_to_one()
    test_class_api()
    print("All tests passed.")