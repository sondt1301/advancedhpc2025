import math
import time
import matplotlib.pyplot as plt
import numpy as np
from numba import cuda
import cv2

# Import image
img = cv2.imread('lenna.png')
img_width = img.shape[1]
img_height = img.shape[0]
print("Original image shape: ", img.shape)

# Copy image to GPU
devOutput = cuda.device_array((img_height, img_width, 3), np.uint8)
devInput = cuda.to_device(img)

# Gaussian filter
gaussian_filter = np.array([
    [0, 0, 1, 2, 1, 0, 0],
    [0, 3, 13, 22, 13, 3, 0],
    [1, 13, 59, 97, 59, 13, 1],
    [2, 22, 97, 159, 97, 22, 2],
    [1, 13, 59, 97, 59, 13, 1],
    [0, 3, 13, 22, 13, 3, 0],
    [0, 0, 1, 2, 1, 0, 0]
])

@cuda.jit
def blur_without_shared_mem(rgb, blur):
    tidx = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    tidy = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y

    radius = 3
    red = 0
    green = 0
    blue = 0
    kernel_sum = 0

    # Avoid out-of-range threads
    if tidx >= img_height or tidy >= img_width:
        return

    for i in range(-radius, radius + 1):
        for j in range(-radius, radius + 1):
            if 0 <= (tidx + i) < img_height and 0 <= (tidy + j) < img_width:
                red += rgb[tidx + i, tidy + j, 0] * gaussian_filter[i + radius, j + radius]
                green += rgb[tidx + i, tidy + j, 1] * gaussian_filter[i + radius, j + radius]
                blue += rgb[tidx + i, tidy + j, 2] * gaussian_filter[i + radius, j + radius]
                kernel_sum += gaussian_filter[i + radius, j + radius]

    blur[tidx, tidy, 0] = red / kernel_sum
    blur[tidx, tidy, 1] = green / kernel_sum
    blur[tidx, tidy, 2] = blue / kernel_sum

blur_without_shared_mem[(math.ceil(img_height / 4), math.ceil(img_width / 4)), (4, 4)](devInput, devOutput)

# # First run to remove odd
# grayscale_gpu[(math.ceil(img_height / 4), math.ceil(img_width / 4)), (4, 4)](devInput, devOutput)
#
# blockSizes = [(4, 4), (8, 8), (16, 16), (32, 32)]
# responseTimes = []
# for blockSize in blockSizes:
#     gridSize = (math.ceil(img_height / blockSize[0]), math.ceil(img_width / blockSize[1]))
#     start_time_gpu = time.time()
#     grayscale_gpu[gridSize, blockSize](devInput, devOutput)
#     end_time_gpu = time.time()
#     gpu_time = end_time_gpu - start_time_gpu
#     responseTimes.append(gpu_time)
#
# print("Response time according to block size: ", responseTimes, blockSizes)
# plt.figure()
# plt.title("Block size vs Time")
# plt.xlabel("Block sizes")
# plt.ylabel("Processing time (s)")
# plt.plot(blockSizes, responseTimes, marker = 'o')
# plt.show()
#
hostOutput = devOutput.copy_to_host()
print("hostOutput shape: ", hostOutput.shape)
hostOutput_converted = np.reshape(hostOutput, (img_width, img_height, 3))

# Show Images
cv2.imshow('Original image', img)
cv2.imshow('Blur image', hostOutput_converted)
cv2.waitKey(0)
cv2.destroyAllWindows()