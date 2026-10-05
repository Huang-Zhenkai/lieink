import numpy as np

from lieink.atoms import Twist
from lieink.containers import LieContainer


def test_lie_container_accepts_iterable_head_and_tail_additions():
    container = LieContainer([Twist(np.ones((6, 1))), Twist(np.ones((6, 1)) * 2)])
    additions = [Twist(np.zeros((6, 1))), Twist(np.ones((6, 1)) * 3)]

    array = container.toNDArray_3D(head_add=additions)

    assert array.shape == (4, 6, 1)
    np.testing.assert_allclose(array[:2], [item.v for item in additions])


def test_lie_container_to_ndarray_formats():
    container = LieContainer([Twist(np.ones((6, 1))), Twist(np.ones((6, 1)) * 2)])

    assert container.toNDArray_2D("r").shape == (12, 1)
    assert container.toNDArray_2D("c").shape == (6, 2)
    assert container.toNDArray_2D("d").shape == (12, 2)
