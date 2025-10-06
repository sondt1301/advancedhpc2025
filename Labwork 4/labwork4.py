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

# Grayscale using CPU
def grayscale_cpu(rgb, gray):
    for i in range(img_height):
        for j in range(img_width):
            gray_value = np.uint8((int(rgb[i, j, 0]) + int(rgb[i, j, 1]) + int(rgb[i, j, 2])) / 3)
            gray[i, j, 0] = gray[i, j, 1] = gray[i, j, 2] = gray_value
    return gray

gray_img_new = np.zeros((img_height, img_width, 3), dtype=np.uint8)
start_time_cpu = time.time()
gray_img_converted = grayscale_cpu(img, gray_img_new)
end_time_cpu = time.time()
cpu_time = end_time_cpu - start_time_cpu
print("Flattened gray image shape: ", gray_img_converted.shape)
print("CPU time in seconds: ", cpu_time)

# Grayscale using GPU
devOutput = cuda.device_array((img_height, img_width, 3), np.uint8)
devInput = cuda.to_device(img)

@cuda.jit
def grayscale_gpu(rgb, gray):
    tidx = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    tidy = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y
    gray_value = np.uint8((rgb[tidx, tidy, 0] + rgb[tidx, tidy, 1] + rgb[tidx, tidy, 2]) / 3)
    gray[tidx, tidy, 0] = gray[tidx, tidy, 1] = gray[tidx, tidy, 2] = gray_value

# First run to remove odd
grayscale_gpu[(math.ceil(img_height / 4), math.ceil(img_width / 4)), (4, 4)](devInput, devOutput)

blockSizes = [(4, 4), (8, 8), (16, 16), (32, 32)]
responseTimes = []
for blockSize in blockSizes:
    gridSize = (math.ceil(img_height / blockSize[0]), math.ceil(img_width / blockSize[1]))
    start_time_gpu = time.time()
    grayscale_gpu[gridSize, blockSize](devInput, devOutput)
    end_time_gpu = time.time()
    gpu_time = end_time_gpu - start_time_gpu
    responseTimes.append(gpu_time)

print("Response time according to block size: ", responseTimes, blockSizes)
plt.figure()
plt.title("Block size vs Time")
plt.xlabel("Block sizes")
plt.ylabel("Processing time (s)")
plt.plot(blockSizes, responseTimes, marker = 'o')
plt.show()

hostOutput = devOutput.copy_to_host()
print("hostOutput shape: ", hostOutput.shape)
hostOutput_converted = np.reshape(hostOutput, (img_width, img_height, 3))

# Show Images
cv2.imshow('Original image', img)
cv2.imshow('CPU gray image', gray_img_converted)
cv2.imshow('CPU gray image', hostOutput_converted)
cv2.waitKey(0)
cv2.destroyAllWindows()