import numpy as np
import sys
import time

def dp(A,B):
    return np.dot(A, B)


arrSize = int(sys.argv[1])
repetitions = int(sys.argv[2])
  
A = np.ones(arrSize,dtype=np.float32)
B = np.ones(arrSize,dtype=np.float32)

halfway = repetitions / 2
total_time = 0
temp = 0.0
for i in range(repetitions):
    start = time.monotonic()
    
    temp = dp(A, B)
    
    end = time.monotonic()
    elapsed_seconds = end - start
    
    if i >= halfway:
        total_time += elapsed_seconds
    
    
avg_time = total_time / halfway

#2 floating point arrays - (2.0 * 4.0 * (arrSize)
bandwidth = ((2.0 * 4.0 * (arrSize)) / (1000000000.0)) / avg_time

#upper bound for dot product computation - (1 addition + 1 multiplication) * (size of floating point arrays) - 1
#actual number of floating point operations for np.dot function probably less due to advanced optimizations
throughput = (2 * (arrSize) - 1) / avg_time
  
print(f"N: {arrSize} <T>: {round(avg_time, 6)} sec B: {round(bandwidth, 3)} GB/sec F: {round(throughput, 3)} FLOP/sec")
print(f"Dot product result: {temp}")
