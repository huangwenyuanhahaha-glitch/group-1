import numpy as np


class Vector:
    def __init__(self, data):
        # Store the numbers in a NumPy array.
        self.data = np.asarray(data, dtype=float)

        if self.data.ndim != 1:
            raise ValueError("Vector data must be one-dimensional")

    def __repr__(self):
        return f"Vector({self.data.tolist()})"


class Matrix:
    def __init__(self, data):
        # Store the numbers in a NumPy array.
        self.data = np.asarray(data, dtype=float)

        if self.data.ndim != 2:
            raise ValueError("Matrix data must be two-dimensional")

    def __repr__(self):
        return f"Matrix({self.data.tolist()})"

    def get_shape(self):
        # Return (number of rows, number of columns).
        return self.data.shape

    def multiply_vector(self, vector):
        # The matrix column count must equal the vector length.
        if self.data.shape[1] != vector.data.shape[0]:
            raise ValueError("Matrix columns must match vector length")

        # NumPy performs the multiplication.
        answer = np.dot(self.data, vector.data)
        return Vector(answer)

    def multiply_matrix(self, other):
        # The first matrix column count must equal the second matrix row count.
        if self.data.shape[1] != other.data.shape[0]:
            raise ValueError("Matrix dimensions do not match for multiplication")

        # NumPy performs the multiplication.
        answer = np.dot(self.data, other.data)
        return Matrix(answer)
