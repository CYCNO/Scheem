import math
import random

from micrograd import Value


class Matrix:
    def __init__(self, rows, cols=None):
        if isinstance(rows, list):
            # If it's a 1D list, wrap it into a 2D list
            if len(rows) > 0 and not isinstance(rows[0], list):
                rows = [rows]

            def wrap_values(data):
                if isinstance(data, list):
                    return [wrap_values(x) for x in data]
                return data if isinstance(data, Value) else Value(data)

            self.matrix = wrap_values(rows)
            self.rows = len(rows)
            self.cols = len(rows[0])
        else:
            self.rows = rows
            self.cols = cols
            self.matrix = [
                [Value(0) for _ in range(self.cols)] for _ in range(self.rows)
            ]

    def fill(self, value=0):
        self.matrix = [
            [Value(value) for _ in range(self.cols)] for _ in range(self.rows)
        ]

    def random(self, lower=0, upper=1):
        self.matrix = [
            [Value(random.uniform(lower, upper)) for _ in range(self.cols)]
            for _ in range(self.rows)
        ]

    @staticmethod
    def sigmoid(A):
        if not isinstance(A, Matrix):
            return NotImplemented
        result = Matrix(A.rows, A.cols)
        for i in range(A.rows):
            for j in range(A.cols):
                result.matrix[i][j] = A.matrix[i][j].sigmoid()
        return result

    @staticmethod
    def relu(A):
        if not isinstance(A, Matrix):
            return NotImplemented
        result = Matrix(A.rows, A.cols)
        for i in range(A.rows):
            for j in range(A.cols):
                result.matrix[i][j] = A.matrix[i][j].relu()
        return result

    @staticmethod
    def add(A, B):
        return A + B

    @staticmethod
    def multiply(A, B):
        return A * B

    def __add__(self, other):
        if not isinstance(other, (Matrix, int, float)):
            return NotImplemented

        if isinstance(other, (int, float)):
            result = Matrix(self.rows, self.cols)
            for i in range(self.rows):
                for j in range(self.cols):
                    result.matrix[i][j] = self.matrix[i][j] + other
            return result

        if self.rows != other.rows or self.cols != other.cols:
            raise ValueError("Matrix dimensions Not Equal for addition")

        result = Matrix(self.rows, self.cols)
        for i in range(self.rows):
            for j in range(self.cols):
                result.matrix[i][j] = self.matrix[i][j] + other.matrix[i][j]
        return result

    def __mul__(self, other):
        if not isinstance(other, (Matrix, int, float)):
            return NotImplemented

        if isinstance(other, (int, float)):
            result = Matrix(self.rows, self.cols)
            for i in range(self.rows):
                for j in range(self.cols):
                    result.matrix[i][j] = self.matrix[i][j] * other
            return result

        if self.cols != other.rows:
            raise ValueError("Columns of first matrix must equal rows of second")

        result = Matrix(self.rows, other.cols)
        for i in range(self.rows):
            for j in range(other.cols):
                val = Value(0.0)
                for k in range(self.cols):
                    val = val + self.matrix[i][k] * other.matrix[k][j]
                result.matrix[i][j] = val
        return result

    def __neg__(self):
        result = Matrix(self.rows, self.cols)
        for i in range(self.rows):
            for j in range(self.cols):
                result.matrix[i][j] = -self.matrix[i][j]
        return result

    def __sub__(self, other):
        if not isinstance(other, (Matrix, int, float)):
            return NotImplemented
        return self + (-other)

    def __rsub__(self, other):
        if not isinstance(other, (Matrix, int, float)):
            return NotImplemented
        return other + (-self)

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    def __getitem__(self, index):
        return self.matrix[index]

    def __repr__(self):
        return "\n".join(str(row) for row in self.matrix)
