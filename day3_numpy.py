import numpy as np

# EXERCISE 1: Create and explore arrays
data = np.array([23, 45, 12, 67, 34, 89, 56, 78, 90, 11])

print("=== Array Stats ===")
print(f"Shape:  {data.shape}")
print(f"Mean:   {data.mean():.2f}")
print(f"Std:    {data.std():.2f}")
print(f"Min/Max: {data.min()} / {data.max()}")


# EXERCISE 2: Boolean masking - find values above average
threshold = data.mean()
above_avg = data[data > threshold]

print(f"\nAbove average ({threshold:.1f}): {above_avg}")


# EXERCISE 3: Matrix operations
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

print("\n=== Matrix Operations ===")
print("A + B =\n", A + B)
print("A @ B (dot product) =\n", A @ B)
print("Transpose A =\n", A.T)


# EXERCISE 4: Broadcasting - normalise each column
X = np.random.randint(10, 100, size=(5, 3))

print("\nOriginal data:\n", X)

X_norm = (X - X.mean(axis=0)) / X.std(axis=0)

print("Normalised (z-score):\n", X_norm.round(3))
# --------------------------------------------------
# VALIDATION TESTS
# --------------------------------------------------

assert data.shape == (10,)
assert data.min() == 11
assert data.max() == 90

# Check matrix operations
assert np.array_equal(A + B, np.array([[6, 8], [10, 12]]))
assert np.array_equal(A @ B, np.array([[19, 22], [43, 50]]))
assert np.array_equal(A.T, np.array([[1, 3], [2, 4]]))

# Check normalized data shape
assert X_norm.shape == X.shape

print("\n=== All validation tests passed! ===")