
import numpy as np

# 1D Array
arr1 = np.array([10, 20, 30, 40])

# 2D Array
arr2 = np.array([
    [1, 2, 3],
    [4, 5, 6]
])

# 3D Array
arr3 = np.array([
    [[1, 2], [3, 4]],
    [[5, 6], [7, 8]]
])

# Print arrays
print("1D Array:")
print(arr1)

print("\n2D Array:")
print(arr2)

print("\n3D Array:")
print(arr3)

# Check shapes
print("\nArray Shapes:")
print("1D Shape:", arr1.shape)
print("2D Shape:", arr2.shape)
print("3D Shape:", arr3.shape)

# Check dimensions
print("\nDimensions:")
print("1D:", arr1.ndim)
print("2D:", arr2.ndim)
print("3D:", arr3.ndim)