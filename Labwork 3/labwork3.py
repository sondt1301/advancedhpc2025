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

# Flatten image
pixelCount = img_width * img_height
flatten_img = np.reshape(img, (pixelCount, 3))
print("Flattened image shape: ", flatten_img.shape)

# Grayscale using CPU
def grayscale_cpu(rgb, gray):
    for i in range(pixelCount):
        gray_value = np.uint8((int(rgb[i, 0]) + int(rgb[i, 1]) + int(rgb[i, 2])) / 3)
        gray[i, 0] = gray[i, 1] = gray[i, 2] = gray_value
    return gray

flatten_gray_img_new = np.zeros((pixelCount, 3), dtype=np.uint8)
start_time_cpu = time.time()
flatten_gray_img_converted = grayscale_cpu(flatten_img, flatten_gray_img_new)
end_time_cpu = time.time()
cpu_time = end_time_cpu - start_time_cpu
print("Flattened gray image shape: ", flatten_gray_img_converted.shape)
print("CPU time in seconds: ", cpu_time)
gray_img_converted = np.reshape(flatten_gray_img_converted, (img_width, img_height, 3))

# Grayscale using GPU
devOutput = cuda.device_array((pixelCount, 3), np.uint8)
devInput = cuda.to_device(flatten_img)

@cuda.jit
def grayscale_gpu(rgb, gray):
    tidx = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    gray_value = np.uint8((rgb[tidx, 0] + rgb[tidx, 1] + rgb[tidx, 2]) / 3)
    gray[tidx, 0] = gray[tidx, 1] = gray[tidx, 2] = gray_value

# First run to remove odd
grayscale_gpu[math.ceil(pixelCount / 32), 32](devInput, devOutput)

blockSizes = [32, 64, 128, 256]
responseTimes = []
for blockSize in blockSizes:
    gridSize = math.ceil(pixelCount / blockSize)
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
cv2.imshow('GPU gray image image', hostOutput_converted)
cv2.waitKey(0)
cv2.destroyAllWindows()