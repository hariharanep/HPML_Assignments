import matplotlib.pyplot as plt
import numpy as np

bandwidth = 30
peak = 200
arithmetic_intensity = peak / bandwidth  

x_axis = np.arange(0, 10.1, 0.1) 
roofline_curve = np.minimum(bandwidth * x_axis, peak)
plt.figure(figsize=(8,6))
plt.plot(x_axis, roofline_curve, label=f"DRAM {bandwidth} GB/s", color="blue")
plt.hlines(peak, xmin=x_axis[0], xmax=x_axis[-1], colors="red", label=f"Peak {peak} GFLOPS")
plt.axvline(arithmetic_intensity, color="green")
plt.text(arithmetic_intensity, 100, f"Arithmetic Intensity = {arithmetic_intensity:.2f}", color="green")

giga = 1e9

c1_1_x = 1306572476.484 / (5.226 * giga)
c1_1_y = 1306572476.484 / giga
plt.scatter(c1_1_x, c1_1_y, color="black")  
plt.annotate("C1_1", xy=(c1_1_x, c1_1_y), 
             fontsize=10)

c1_2_x = 1276139060.669 / (5.105 * giga)
c1_2_y = 1276139060.669 / giga
plt.scatter(c1_2_x, c1_2_y, color="black")  
plt.annotate("C1_2", xy=(c1_2_x, c1_2_y), 
             fontsize=10)

c2_1_x = 2379464971.733 / (9.518 * giga)
c2_1_y = 2379464971.733 / giga
plt.scatter(c2_1_x, c2_1_y, color="black")  
plt.annotate("C2_1", xy=(c2_1_x, c2_1_y), 
             fontsize=10)

c2_2_x = 2542291554.957 / (10.169 * giga)
c2_2_y = (2542291554.957 / giga)
plt.scatter(c2_2_x, c2_2_y, color="black")  
plt.annotate("C2_2", xy=(c2_2_x, c2_2_y), 
             fontsize=10)

c3_1_x = 23323663337.679 / (93.295 * giga)
c3_1_y = 23323663337.679 / giga
plt.scatter(c3_1_x, c3_1_y, color="black")  
plt.annotate("C3_1", xy=(c3_1_x, c3_1_y), 
             fontsize=10)

c3_2_x = 11232733640.898 / (44.931 * giga)
c3_2_y = 11232733640.898 / giga
plt.scatter(c3_2_x, c3_2_y, color="black")  
plt.annotate("C3_2", xy=(c3_2_x, c3_2_y), 
             fontsize=10)

c4_1_x = 7219940.395 / (0.029 * giga)
c4_1_y = 7219940.395 / giga
plt.scatter(c4_1_x, c4_1_y, color="black")  
plt.annotate("C4_1", xy=(c4_1_x, c4_1_y), 
             fontsize=10)

c4_2_x = 6351966.144 / (0.025 * giga)
c4_2_y = 6351966.144 / giga
plt.scatter(c4_2_x, c4_2_y, color="black")  
plt.annotate("C4_2", xy=(c4_2_x, c4_2_y), 
             fontsize=10)

c5_1_x = 6007038847.873 / (24.028 * giga)
c5_1_y = 6007038847.873 / giga
plt.scatter(c5_1_x, c5_1_y, color="black")  
plt.annotate("C5_1", xy=(c5_1_x, c5_1_y), 
             fontsize=10)

c5_2_x = 3124080208.627 / (12.496 * giga)
c5_2_y = 3124080208.627 / giga
plt.scatter(c5_2_x, c5_2_y, color="black")  
plt.annotate("C5_2", xy=(c5_2_x, c5_2_y), 
             fontsize=10)

plt.xlabel("Arithmetic Intensity [FLOP/bytes]")
plt.ylabel("Actual FLOPS")
plt.title("Roofline Model")
plt.legend()
plt.xscale("log")
plt.yscale("log")
plt.grid(True, which="both", lw=0.5)
plt.show()

print((c1_1_x, c1_1_y))
print((c1_2_x, c1_2_y))
print((c2_1_x, c2_1_y))
print((c2_2_x, c2_2_y))
print((c3_1_x, c3_1_y))
print((c3_2_x, c3_2_y))
print((c4_1_x, c4_1_y))
print((c4_2_x, c4_2_y))
print((c5_1_x, c5_1_y))
print((c5_2_x, c5_2_y))
