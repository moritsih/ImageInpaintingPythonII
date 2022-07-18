# ImageInpaintingPythonII
Programming in Python II @ JKU Linz

The task was to create and train a convolutional neural network that would predict the missing parts images where a grid of pixels had been set to zero. 
The images had been preprocessed by setting an offset (distance from the border) and a spacing (distance between grid lattices) that would be applied
to set pixels to zero. 
An offset of, e. g., 8 would result in the first 8 rows and columns (left and top) to be zero. Spacing of, e. g., 2, means one row and column of pixels
is followed by one row and column of zeros.
