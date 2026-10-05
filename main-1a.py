import numpy as np 

arr = np.array([1,2,3,4,5,6])
print("1D Array : ", arr)

# reshape 1D array into 2D array with 2 rows and 3 columns
arr2d = arr.reshape(2, 3)
print("2D Array :\n", arr2d)

# slice the 2D array: first row, all columns
slice_row = arr2d[0, :]
print("First row slice :", slice_row)

# slice the 2D array: all rows, last two columns
slice_cols = arr2d[:, 1:]
print("Last two columns slice :\n", slice_cols)

 