
import numpy as np

# 1. Broadcasting
print("===== BROADCASTING =====")

a = np.array([10, 20, 30])
result = a + 5

print("Original Array:", a)
print("After Adding 5:", result)


# 2. Vectorised Operations
print("\n===== VECTORIZED OPERATIONS =====")

x = np.array([1, 2, 3])
y = np.array([4, 5, 6])

addition = x + y
multiplication = x * y

print("Addition:", addition)
print("Multiplication:", multiplication)


# 3. Matrix Multiplication
print("\n===== MATRIX MULTIPLICATION =====")

matrix1 = np.array([
    [1, 2],
    [3, 4]
])

matrix2 = np.array([
    [5, 6],
    [7, 8]
])

result = np.matmul(matrix1, matrix2)

print("Matrix 1:")
print(matrix1)

print("Matrix 2:")
print(matrix2)

print("Multiplication Result:")
print(result)