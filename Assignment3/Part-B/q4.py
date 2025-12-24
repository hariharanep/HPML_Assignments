import matplotlib.pyplot as plt

# Initialize Data for plottingg
k = [1, 5, 10, 50, 100]
cuda_1_1_1 = [0.113437, 0.300549, 0.602630, 3.012353, 6.021396]
cuda_256_1 = [0.002198, 0.010550, 0.019459, 0.086969, 0.134431]
cuda_256_3 = [0.000720, 0.003510, 0.007036, 0.035145, 0.066698]
cuda_um_1_1_1 = [0.086428, 0.300575, 0.602516, 3.011691, 6.022818]
cuda_um_256_1 = [0.002123, 0.010118, 0.019027, 0.074960, 0.146477]
cuda_um_256_3 = [0.000747, 0.003276, 0.006785, 0.037977, 0.058270]

cpu = [0.002749, 0.013780, 0.027743, 0.139319, 0.277253]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Create Plot for using cuda kernel without unified memory
ax1.plot(k, cuda_1_1_1, label='1 thread, 1 block')
ax1.plot(k, cuda_256_1, label='256 threads, 1 block')
ax1.plot(k, cuda_256_3, label='256 threads, 3 blocks')
ax1.plot(k, cpu, label='CPU')
ax1.set_xlabel('K(Size of each array in millions)')
ax1.set_ylabel('Time(Seconds)')
ax1.set_title('Plot of Q2 Results')
ax1.legend()
ax1.set_yscale('log')

# Create Plot for using cuda kernel with unified memory
ax2.plot(k, cuda_um_1_1_1, label='1 thread, 1 block')
ax2.plot(k, cuda_um_256_1, label='256 threads, 1 block')
ax2.plot(k, cuda_um_256_3, label='256 threads, 3 blocks')
ax2.plot(k, cpu, label='CPU')
ax2.set_xlabel('K(Size of each array in millions)')
ax2.set_ylabel('Time(Seconds)')
ax2.set_title('Plot of Q3 Results')
ax2.legend()
ax2.set_yscale('log')

plt.show()