import numpy as np
import matplotlib.pyplot as plt
from numba import cuda
import matplotlib.animation as animation

# Grid size and simulation steps
N = 256
steps = 200
TPB = 16

# CUDA kernel
@cuda.jit
def heat_diffusion_kernel(current, next):
    x, y = cuda.grid(2)
    if 1 <= x < current.shape[0] - 1 and 1 <= y < current.shape[1] - 1:
        next[x, y] = 0.25 * (current[x+1, y] + current[x-1, y] +
                             current[x, y+1] + current[x, y-1])

# Initialize heat grid
grid = np.zeros((N, N), dtype=np.float32)
grid[N//2 - 10:N//2 + 10, N//2 - 10:N//2 + 10] = 100.0  # Heat source

# Allocate GPU memory
d_current = cuda.to_device(grid)
d_next = cuda.device_array_like(d_current)

# CUDA thread/block setup
blocks_per_grid = (N // TPB, N // TPB)
threads_per_block = (TPB, TPB)

# Prepare matplotlib figure
fig, ax = plt.subplots()
img = ax.imshow(grid, cmap='hot', interpolation='nearest', vmin=0, vmax=100)
plt.title("2D Heat Diffusion")

# This function will update the frame for each step
def update(frame):
    global d_current, d_next
    heat_diffusion_kernel[blocks_per_grid, threads_per_block](d_current, d_next)
    d_current, d_next = d_next, d_current
    img.set_data(d_current.copy_to_host())
    return [img]

# Create animation
ani = animation.FuncAnimation(fig, update, frames=steps, interval=30, blit=True)
plt.show()
