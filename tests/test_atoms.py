import numpy as np

from lieink.atoms import Point4, SO3, Twist


def test_point4_addition_returns_normalized_homogeneous_point():
    first = Point4(np.array([[1.0], [2.0], [3.0], [2.0]]))
    second = Point4(np.array([[2.0], [3.0], [4.0], [4.0]]))

    result = first + second

    np.testing.assert_allclose(result.v, [[1.0], [1.75], [2.5], [1.0]])


def test_point4_vector4_conversion_preserves_direction_and_vector_marker():
    point = Point4(np.array([[1.0], [2.0], [3.0], [2.0]]))

    result = point.get_Vector4()

    np.testing.assert_allclose(result.v, [[0.5], [1.0], [1.5], [0.0]])


def test_so3_rotation_log_roundtrip():
    angle = 0.7
    rotation = SO3(SO3.rotzc(angle))

    logarithm = rotation.log("so3")

    np.testing.assert_allclose(logarithm.v, [[0.0, -angle, 0.0], [angle, 0.0, 0.0], [0.0, 0.0, 0.0]])


def test_twist_exp_log_roundtrip():
    value = np.arange(6.0).reshape(6, 1) / 3.0
    recovered = Twist(value).exp(1.0).log("Twist")

    np.testing.assert_allclose(recovered.v, value, atol=1e-6)
