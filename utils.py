"""
Author: Moritz Haderer
Matr.Nr.: K11774793
Exercise 5
"""

import numpy as np

def set_random_spacing_offset(SEED):
    np.random.seed(SEED)

    #offsets = [0,1,2,3,4,5,6,7,8]
    #spacings = [2,3,4,5,6]

    return (np.random.randint(0,8), np.random.randint(0,8)), \
           (np.random.randint(2,6), np.random.randint(2,6))

'''
#for i in range(3):
arrays, idx = dataset[4]
input_array, known_array, target_array = arrays
print(input_array)
print(f"{len(target_array)}")
print(f"known_array = 0 mask: {len(np.ravel(input_array[known_array==0]))}")
input_array[known_array == 0] = target_array
input_array = np.moveaxis(input_array, 0, 2)
PIL_image = Image.fromarray(input_array.astype('uint8'), 'RGB')
plt.imshow(PIL_image)
#plt.show()
'''
