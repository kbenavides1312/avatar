import dash
from dash import dcc, html, Input, Output, callback_context, State
import plotly.graph_objects as go
import numpy as np
from numba import cuda
import math
import time
import threading
import queue
from flask import Flask, request, jsonify
from flask_socketio import SocketIO, emit

# Simulation parameters
WIDTH, HEIGHT = 1200, 400
NUM_PARTICLES = 10000  # Reduced for web performance
TPB = 32
DT = 0.05
GRAVITY = 9.8e2
DAMPING = 0.1
RADIUS = 3.0
PULL_STRENGTH = -30.0e2
FORCE_CIRCLE_SIZE = 50
FORCE_DECAY_FACTOR = 1e0
VISCOSITY = 0.8
rect_width, rect_height = 20, 80
RECT_FORCE = np.array([0, PULL_STRENGTH], dtype=np.float32)

# Global variables for simulation state
positions = np.zeros((NUM_PARTICLES, 2), dtype=np.float32)
positions[:, 0] = np.random.uniform(0, WIDTH, NUM_PARTICLES)
positions[:, 1] = np.random.uniform(HEIGHT // 2, HEIGHT - 50, NUM_PARTICLES)
velocities = np.zeros((NUM_PARTICLES, 2), dtype=np.float32)

circle_center = [0, 0]
rect_data = np.array([WIDTH//2 - rect_width//2, HEIGHT//2 - rect_height//2, rect_width, rect_height], dtype=np.float32)

# CUDA device arrays
d_pos = cuda.to_device(positions)
d_vel = cuda.to_device(velocities)
d_rect = cuda.to_device(rect_data)
d_target = cuda.to_device(np.array([0.0, 0.0], dtype=np.float32))
d_attract = cuda.to_device(np.array([1], dtype=np.float32))  # Start with attraction enabled

# Thread-safe queue for communication between simulation and web server
data_queue = queue.Queue(maxsize=10)
simulation_running = True
connected_clients = set()

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
        inside_rect = rect_x <= pos[i, 0] <= rect_x + rect_w and rect_y <= pos[i, 1] <= rect_y + rect_h
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

            center_x = (rect_x + rect_w) / 2
            center_y = (rect_y + rect_h) / 2
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
                # Apply pull proportional to distance
                force = math.sqrt(pull_strength[0]**2 + pull_strength[1]**2)
                if nx:
                    force_x = force * math.exp(- FORCE_DECAY_FACTOR * (dist - FORCE_CIRCLE_SIZE) ** 2)
                    vel[i, 0] += nx * force_x * dt
                if ny:
                    force_y = pull_strength[1] * math.exp(- FORCE_DECAY_FACTOR * (dist - FORCE_CIRCLE_SIZE) ** 2)
                    vel[i, 1] += ny * force_y * dt

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

    # Particle-particle collisions (simplified for performance)
    for j in range(i + 1, pos.shape[0]):
        dx = pos[i, 0] - pos[j, 0]
        dy = pos[i, 1] - pos[j, 1]
        dist = math.sqrt(dx * dx + dy * dy)
        if dist < 2 * radius and dist > 1e-5:
            # Elastic collision
            vel_i_x = vel[i, 0]
            vel_i_y = vel[i, 1]
            vel[i, 0] = vel[j, 0] * VISCOSITY
            vel[i, 1] = vel[j, 1] * VISCOSITY
            vel[j, 0] = vel_i_x * VISCOSITY
            vel[j, 1] = vel_i_y * VISCOSITY

            # Separate particles
            overlap = 2 * radius - dist
            norm_x = dx / dist
            norm_y = dy / dist
            pos[i, 0] += 0.5 * overlap * norm_x
            pos[i, 1] += 0.5 * overlap * norm_y
            pos[j, 0] -= 0.5 * overlap * norm_x
            pos[j, 1] -= 0.5 * overlap * norm_y

def simulation_thread():
    """Run the CUDA simulation in a separate thread"""
    global positions, velocities, rect_data
    blocks = (NUM_PARTICLES + TPB - 1) // TPB
    
    while simulation_running:
        start_time = time.time()
        
        # Run CUDA kernel
        update[blocks, TPB](d_pos, d_vel, DT, GRAVITY, DAMPING, RADIUS, WIDTH, HEIGHT, 
                           d_target, d_attract, d_rect, RECT_FORCE)
        cuda.synchronize()
        
        # Copy data back to host
        positions = d_pos.copy_to_host()
        
        # Put data in queue for web server
        try:
            data_queue.put_nowait({
                'positions': positions.copy().tolist(),
                'rect_data': {
                    'x': float(rect_data[0]),
                    'y': float(rect_data[1]),
                    'width': float(rect_data[2]),
                    'height': float(rect_data[3])
                },
                'timestamp': time.time()
            })
        except queue.Full:
            # Skip frame if queue is full
            pass
        
        # Control frame rate
        elapsed = time.time() - start_time
        if elapsed < 0.033:  # ~30 FPS for web
            time.sleep(0.033 - elapsed)

# Initialize Dash app
app = dash.Dash(__name__)
server = app.server

# Initialize SocketIO for React app
socketio = SocketIO(server, cors_allowed_origins="*")

# App layout
app.layout = html.Div([
    html.H1("Particle Simulation Stream", style={'textAlign': 'center'}),
    
    html.Div([
        html.P("Click and drag to move the force rectangle", style={'textAlign': 'center'}),
        html.P("Particles will be attracted to the rectangle", style={'textAlign': 'center'})
    ], style={'marginBottom': '20px'}),
    
    dcc.Graph(
        id='particle-plot',
        style={'height': '600px'},
        config={'displayModeBar': False}
    ),
    
    dcc.Interval(
        id='interval-component',
        interval=33,  # ~30 FPS
        n_intervals=0
    ),
    
    html.Div([
        html.Button('Start Simulation', id='start-btn', n_clicks=0),
        html.Button('Stop Simulation', id='stop-btn', n_clicks=0),
        html.Button('Reset Particles', id='reset-btn', n_clicks=0),
    ], style={'textAlign': 'center', 'marginTop': '20px'}),
    
    # Hidden div to store rectangle position
    dcc.Store(id='rect-store', data={'x': rect_data[0], 'y': rect_data[1]}),
    
    # Hidden div for mouse events
    html.Div(id='mouse-events', style={'display': 'none'})
])

@app.callback(
    Output('particle-plot', 'figure'),
    Input('interval-component', 'n_intervals'),
    Input('start-btn', 'n_clicks'),
    Input('stop-btn', 'n_clicks'),
    Input('reset-btn', 'n_clicks'),
    State('rect-store', 'data')
)
def update_graph(n_intervals, start_clicks, stop_clicks, reset_clicks, rect_store):
    global simulation_running, positions, velocities, rect_data
    
    # Handle start/stop buttons
    ctx = callback_context
    if ctx.triggered:
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
        if button_id == 'start-btn':
            simulation_running = True
        elif button_id == 'stop-btn':
            simulation_running = False
        elif button_id == 'reset-btn':
            # Reset particles
            positions[:, 0] = np.random.uniform(0, WIDTH, NUM_PARTICLES)
            positions[:, 1] = np.random.uniform(HEIGHT // 2, HEIGHT - 50, NUM_PARTICLES)
            velocities = np.zeros((NUM_PARTICLES, 2), dtype=np.float32)
            d_pos.copy_to_device(positions)
            d_vel.copy_to_device(velocities)
    
    # Update rectangle position if changed
    if rect_store and (rect_store['x'] != rect_data[0] or rect_store['y'] != rect_data[1]):
        rect_data[0] = rect_store['x']
        rect_data[1] = rect_store['y']
        d_rect.copy_to_device(rect_data)
    
    # Get latest data from simulation thread
    try:
        data = data_queue.get_nowait()
        positions = data['positions']
        rect_data = data['rect_data']
    except queue.Empty:
        # Use last known positions if no new data
        pass
    
    # Create scatter plot
    fig = go.Figure()
    
    # Add particles
    fig.add_trace(go.Scatter(
        x=positions[:, 0],
        y=positions[:, 1],
        mode='markers',
        marker=dict(
            size=2,
            color='blue',
            opacity=0.7
        ),
        name='Particles',
        showlegend=False
    ))
    
    # Add rectangle
    fig.add_trace(go.Scatter(
        x=[rect_data[0], rect_data[0] + rect_data[2], rect_data[0] + rect_data[2], rect_data[0], rect_data[0]],
        y=[rect_data[1], rect_data[1], rect_data[1] + rect_data[3], rect_data[1] + rect_data[3], rect_data[1]],
        mode='lines',
        fill='toself',
        fillcolor='rgba(0, 0, 0, 0.5)',
        line=dict(color='black', width=1),
        name='Force Rectangle',
        showlegend=False
    ))
    
    # Update layout
    fig.update_layout(
        title='Real-time Particle Simulation',
        xaxis=dict(
            range=[0, WIDTH],
            showgrid=False,
            zeroline=False,
            showticklabels=False
        ),
        yaxis=dict(
            range=[0, HEIGHT],
            showgrid=False,
            zeroline=False,
            showticklabels=False
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(l=0, r=0, t=50, b=0),
        height=600,
        # Add click and drag events
        dragmode='pan',
        clickmode='event'
    )
    
    return fig

@app.callback(
    Output('rect-store', 'data'),
    Input('particle-plot', 'clickData'),
    Input('particle-plot', 'relayoutData'),
    State('rect-store', 'data')
)
def update_rect_position(click_data, relayout_data, current_rect):
    global rect_data
    
    if not current_rect:
        current_rect = {'x': rect_data[0], 'y': rect_data[1]}
    
    # Handle click events
    if click_data:
        point = click_data['points'][0]
        x, y = point['x'], point['y']
        # Move rectangle to clicked position (center it)
        current_rect['x'] = x - rect_width // 2
        current_rect['y'] = y - rect_height // 2
    
    # Handle drag events
    if relayout_data and 'xaxis.range[0]' in relayout_data:
        # This is a pan event, not a drag event
        pass
    
    return current_rect

# Add Flask route for external updates
@server.route('/update_rect', methods=['POST'])
def update_rect():
    global rect_data
    data = request.get_json()
    x = data.get('x', rect_data[0])
    y = data.get('y', rect_data[1])
    
    # Update rectangle position
    rect_data[0] = x - rect_width // 2
    rect_data[1] = y - rect_height // 2
    d_rect.copy_to_device(rect_data)
    
    return jsonify({'status': 'success'})

# SocketIO event handlers for React app
@socketio.on('connect')
def handle_connect():
    print(f'React client connected: {request.sid}')
    connected_clients.add(request.sid)
    emit('connected', {'status': 'connected'})

@socketio.on('disconnect')
def handle_disconnect():
    print(f'React client disconnected: {request.sid}')
    connected_clients.discard(request.sid)

@socketio.on('start_simulation')
def handle_start_simulation():
    global simulation_running
    simulation_running = True
    emit('simulation_started', {'status': 'started'})

@socketio.on('stop_simulation')
def handle_stop_simulation():
    global simulation_running
    simulation_running = False
    emit('simulation_stopped', {'status': 'stopped'})

@socketio.on('reset_simulation')
def handle_reset_simulation():
    global positions, velocities
    positions[:, 0] = np.random.uniform(0, WIDTH, NUM_PARTICLES)
    positions[:, 1] = np.random.uniform(HEIGHT // 2, HEIGHT - 50, NUM_PARTICLES)
    velocities = np.zeros((NUM_PARTICLES, 2), dtype=np.float32)
    d_pos.copy_to_device(positions)
    d_vel.copy_to_device(velocities)
    emit('simulation_reset', {'status': 'reset'})

@socketio.on('update_rect')
def handle_update_rect(data):
    global rect_data
    rect_data[0] = data['x']
    rect_data[1] = data['y']
    d_rect.copy_to_device(rect_data)

def broadcast_simulation_data():
    """Broadcast simulation data to all connected React clients"""
    while True:
        try:
            data = data_queue.get(timeout=1)
            if connected_clients:
                socketio.emit('simulation_data', data, room=None)
        except queue.Empty:
            continue
        except Exception as e:
            print(f"Error broadcasting data: {e}")

# Start simulation and broadcast threads
sim_thread = threading.Thread(target=simulation_thread, daemon=True)
broadcast_thread = threading.Thread(target=broadcast_simulation_data, daemon=True)

if __name__ == '__main__':
    print("Starting combined Dash + SocketIO server...")
    print("Dash app will be available at: http://127.0.0.1:8050")
    print("React app should connect to: http://127.0.0.1:8050")
    
    sim_thread.start()
    broadcast_thread.start()
    
    socketio.run(server, debug=True, host='0.0.0.0', port=8050) 