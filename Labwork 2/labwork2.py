from numba import cuda
device = cuda.select_device(0)
multiprocessor_count = device.MULTIPROCESSOR_COUNT
core_count = 64 * multiprocessor_count
free_memory, total_memory = cuda.current_context().get_memory_info()
free_memory_mb = free_memory / (1024 * 1024)
total_memory_mb = total_memory / (1024 * 1024)

print("===== Device name ====")
print(device.name)

print("===== Core information ====")
print("Multiprocessor count: ", multiprocessor_count)
print("Device compatibility: ", device.compute_capability)
print("Core count: ", core_count)

print("===== Memory information ====")
print("Free memory in MB: ", free_memory_mb)
print("Total memory in MB: ", total_memory_mb)

