import numpy as np
import pytest

from core import Matrix, Vector


def test_vector_stores_numbers():
    vector = Vector([1, 2, 3])

    assert isinstance(vector.data, np.ndarray)
    assert vector.data.tolist() == [1.0, 2.0, 3.0]


def test_matrix_stores_numbers_and_shape():
    matrix = Matrix([[1, 2], [3, 4]])

    assert isinstance(matrix.data, np.ndarray)
    assert matrix.data.tolist() == [[1.0, 2.0], [3.0, 4.0]]
    assert matrix.get_shape() == (2, 2)


def test_vector_rejects_two_dimensional_data():
    with pytest.raises(ValueError, match="one-dimensional"):
        Vector([[1, 2], [3, 4]])


def test_matrix_rejects_one_dimensional_data():
    with pytest.raises(ValueError, match="two-dimensional"):
        Matrix([1, 2, 3])


def test_matrix_vector_multiplication():
    matrix = Matrix([[1, 2], [3, 4]])
    vector = Vector([5, 6])

    result = matrix.multiply_vector(vector)

    assert isinstance(result, Vector)
    assert result.data.tolist() == [17.0, 39.0]


def test_matrix_matrix_multiplication():
    first = Matrix([[1, 2], [3, 4]])
    second = Matrix([[5, 6], [7, 8]])

    result = first.multiply_matrix(second)

    assert isinstance(result, Matrix)
    assert result.data.tolist() == [[19.0, 22.0], [43.0, 50.0]]


def test_matrix_vector_multiplication_rejects_wrong_dimensions():
    matrix = Matrix([[1, 2], [3, 4]])
    vector = Vector([1, 2, 3])

    with pytest.raises(ValueError, match="columns must match"):
        matrix.multiply_vector(vector)


def test_matrix_matrix_multiplication_rejects_wrong_dimensions():
    first = Matrix([[1, 2], [3, 4]])
    second = Matrix([[1, 2]])

    with pytest.raises(ValueError, match="dimensions do not match"):
        first.multiply_matrix(second)
