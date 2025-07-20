import matplotlib.patches as patches
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from numba import cuda, float32
import math
import time

# Simulation parameters
WIDTH, HEIGHT = 1200, 400
NUM_PARTICLES = 10000
TPB = 32
DT = 0.05
GRAVITY = -9.8e2
DAMPING = 0.1
RADIUS = 3.0
PULL_STRENGTH = 30.0e2  # tuning parameter for attraction force
FORCE_CIRCLE_SIZE = 50
FORCE_DECAY_FACTOR = 1e0
VISCOSITY = 0.8
rect_width, rect_height = 20, 80
RECT_FORCE = np.array([0, PULL_STRENGTH], dtype=np.float32)

positions = np.zeros((NUM_PARTICLES, 2), dtype=np.float32)
positions[:, 0] = np.random.uniform(0, WIDTH, NUM_PARTICLES)
positions[:, 1] = np.random.uniform(HEIGHT // 2, HEIGHT - 50, NUM_PARTICLES)
velocities = np.zeros((NUM_PARTICLES, 2), dtype=np.float32)

circle_center = [0, 0]
rect_data = np.array([0, 0, rect_width, rect_height], dtype=np.float32)

d_pos = cuda.to_device(positions)
d_vel = cuda.to_device(velocities)
d_rect = cuda.to_device(rect_data)

# Use device arrays to pass target and attract flag
d_target = cuda.to_device(np.array([0.0, 0.0], dtype=np.float32))
d_attract = cuda.to_device(np.array([0], dtype=np.int32))  # 1 if attract active, else 0

@cuda.jit
def update(pos, vel, dt, gravity, damping, radius, width, height, target, attract, rect, pull_strength):
    i = cuda.grid(1)
    if i >= pos.shape[0]:
        return

    # Gravity pulls down (y increases downward)
    vel[i, 1] += gravity * dt

    # If attract flag set, apply pulling force towards target
    if attract[0] == 1:
        rect_x = rect[0]
        rect_y = rect[1]
        rect_w = rect[2]
        rect_h = rect[3]

        # rect power
        inside_rect = rect_x <= pos[i, 0] <= rect_x + rect_width and rect_y <= pos[i, 1] <= rect_y + rect_height
        if inside_rect:
            force_x = pull_strength[0]
            force_y = pull_strength[1]
            vel[i, 0] += force_x * dt
            vel[i, 1] += force_y * dt
        else:
            # Distance to each edge
            dist_top    = abs(pos[i, 1] - rect_y)
            dist_bottom = abs(pos[i, 1] - (rect_y + rect_h))
            dist_left   = abs(pos[i, 0] - rect_x)
            dist_right  = abs(pos[i, 0] - (rect_x + rect_w))

            # Find minimum
            min_dist = min(dist_top, dist_bottom, dist_left, dist_right)

            center_x = (rect_x+rect_width)/2
            center_y = (rect_y+rect_height)/2
            dx = pos[i, 0] - center_x 
            dy = pos[i, 1] - center_y
            if min_dist == dist_top:
                dist = abs(dy)
                dx = 0
            elif min_dist == dist_bottom:
                dist = abs(dy)
                dx = 0
            elif min_dist == dist_left:
                dist = abs(dx)
                dy = 0
            else:
                dist = abs(dx)
                dy = 0
            if dist > 1e-5:
                # Normalize vector
                nx = dx / dist
                ny = dy / dist
                # Apply pull proportional to distance (could tweak)
                force = math.sqrt(pull_strength[0]**2 + pull_strength[1]**2)
                if nx:
                    force_x = force * math.exp(- FORCE_DECAY_FACTOR * (dist - FORCE_CIRCLE_SIZE) ** 2)  # stronger when closer
                    vel[i, 0] += nx * force_x * dt
                if ny:
                    force_y = pull_strength[1] * math.exp(- FORCE_DECAY_FACTOR * (dist - FORCE_CIRCLE_SIZE) ** 2)  # stronger when closer
                    vel[i, 1] += ny * force_y * dt

        # # circle power
        # dx = target[0] - pos[i, 0]
        # dy = target[1] - pos[i, 1]
        # dist = math.sqrt(dx*dx + dy*dy)
        # if dist > 1e-5:
        #     # Normalize vector
        #     nx = dx / dist
        #     ny = dy / dist
        #     # Apply pull proportional to distance (could tweak)
        #     force = pull_strength * math.exp(- FORCE_DECAY_FACTOR * (dist - FORCE_CIRCLE_SIZE) ** 2)  # stronger when closer
        #     # Add to velocity
        #     vel[i, 0] += nx * force * dt
        #     vel[i, 1] += ny * force * dt

    # Update position
    pos[i, 0] += vel[i, 0] * dt
    pos[i, 1] += vel[i, 1] * dt

    # Wall collisions
    if pos[i, 0] < radius:
        pos[i, 0] = radius
        vel[i, 0] *= -damping
    if pos[i, 0] > width - radius:
        pos[i, 0] = width - radius
        vel[i, 0] *= -damping
    if pos[i, 1] < radius:
        pos[i, 1] = radius
        vel[i, 1] *= -damping
    if pos[i, 1] > height - radius:
        pos[i, 1] = height - radius
        vel[i, 1] *= -damping

    # Particle-particle collisions (naive O(N^2))
    for j in range(pos.shape[0]):
        if i == j:
            continue
        dx = pos[i, 0] - pos[j, 0]
        dy = pos[i, 1] - pos[j, 1]
        dist = math.sqrt(dx * dx + dy * dy)
        if dist < 2 * radius and dist > 1e-5:
            # Elastic collision (swap velocities)
            vel_i_x = vel[i, 0]
            vel_i_y = vel[i, 1]
            vel[i, 0] = vel[j, 0] * VISCOSITY
            vel[i, 1] = vel[j, 1] * VISCOSITY
            vel[j, 0] = vel_i_x * VISCOSITY
            vel[j, 1] = vel_i_y * VISCOSITY

            # Separate particles so they don't stick
            overlap = 2 * radius - dist
            norm_x = dx / dist
            norm_y = dy / dist
            pos[i, 0] += 0.5 * overlap * norm_x
            pos[i, 1] += 0.5 * overlap * norm_y
            pos[j, 0] -= 0.5 * overlap * norm_x
            pos[j, 1] -= 0.5 * overlap * norm_y


# Setup plot
fig, ax = plt.subplots(figsize=(6, 6))
fig.patch.set_facecolor('white')
ax.set_facecolor('white')

scat = ax.scatter([], [], s=40, c='blue')
ax.set_xlim(0, WIDTH)
ax.set_ylim(0, HEIGHT)
ax.set_aspect('equal', 'box')
ax.axis('off')

circle = patches.Circle((0, 0), FORCE_CIRCLE_SIZE, fill=False, edgecolor='red', linewidth=5)
ax.add_patch(circle)
# Hide circle initially
circle.set_visible(False)

rect = patches.Rectangle((rect_data[0], rect_data[1]), rect_width, rect_height, fill=False, edgecolor='green', linewidth=2)
ax.add_patch(rect)
rect.set_visible(True)

blocks = (NUM_PARTICLES + TPB - 1) // TPB

def on_press(event):
    global rect_data
    if event.inaxes != ax:
        return
    if event.xdata is None or event.ydata is None:
        return
    # Update attract flag and target position
    circle_center[0] = event.xdata
    circle_center[1] = event.ydata
    # circle.set_visible(True)
    
    ll_x = event.xdata - rect_width // 2
    ll_y = event.ydata - rect_height // 2
    rect.set_xy((ll_x, ll_y))
    # rect.set_visible(True)
    fig.canvas.draw_idle()
    rect_data[:] = [ll_x, ll_y, rect_width, rect_height]
    d_rect.copy_to_device(rect_data)
    d_attract.copy_to_device(np.array([1], dtype=np.int32))
    d_target.copy_to_device(np.array([event.xdata, event.ydata], dtype=np.float32))

def on_release(event):
    # Disable attract force
    # circle.set_visible(False)
    ll_x = event.xdata - rect_width // 2
    ll_y = event.ydata - rect_height // 2
    rect.set_xy((ll_x, ll_y))
    fig.canvas.draw_idle()
    rect_data[:] = [ll_x, ll_y, rect_width, rect_height]
    d_rect.copy_to_device(rect_data)
    print(f"rect data {rect_data[0]} {rect_data[1]} {rect_data[2]} {rect_data[3]}")
    # d_attract.copy_to_device(np.array([0], dtype=np.int32))

def on_move(event):
    # If attract active, update target position as mouse moves
    return
    if event.xdata is None or event.ydata is None:
        return
    if d_attract.copy_to_host()[0] == 1 and event.inaxes == ax:
        circle_center[0] = event.xdata
        circle_center[1] = event.ydata
        # fig.canvas.draw_idle()
        d_target.copy_to_device(np.array([event.xdata, event.ydata], dtype=np.float32))

fig.canvas.mpl_connect('button_press_event', on_press)
fig.canvas.mpl_connect('button_release_event', on_release)
fig.canvas.mpl_connect('motion_notify_event', on_move)

def animate(_):
    global rect_data
    start = time.time()
    update[blocks, TPB](d_pos, d_vel, DT, GRAVITY, DAMPING, RADIUS, WIDTH, HEIGHT, d_target, d_attract, d_rect, RECT_FORCE)
    cuda.synchronize()

    pos_host = d_pos.copy_to_host()
    scat.set_offsets(pos_host)
    rect.set_xy((rect_data[0], rect_data[1]))

    circle.center = (circle_center[0], circle_center[1])
    
    # Update rectangle position to match the current rect_data
    # if rect.get_visible():
    #     # print(f"rect data {rect_data[0]} {rect_data[1]} {rect_data[2]} {rect_data[3]}")
    # if rect.get_x() != rect_data[0]:
    #     rect.set_xy((rect_data[0], rect_data[1]))
    #     fig.canvas.draw_idle()
    
    end = time.time()
    # print(f"Frame time: {(end - start) * 1000:.2f} ms")
    return scat, rect

ani = animation.FuncAnimation(fig, animate, interval=5, blit=True)
plt.show()
