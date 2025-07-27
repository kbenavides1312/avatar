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
import base64
from PIL import Image, ImageDraw

# Simulation parameters
WIDTH, HEIGHT = 1200, 400
NUM_PARTICLES = 3000  # Reduced for web performance
TPB = 32
DT = 0.05
GRAVITY = 9.8e2
DAMPING = 0.1
RADIUS = 5.0
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

# Store exactly 2 players' force maps
player1_force_map = None
player2_force_map = None
player1_id = None
player2_id = None

# CUDA device arrays
d_pos = cuda.to_device(positions)
d_vel = cuda.to_device(velocities)
d_target = cuda.to_device(np.array([0.0, 0.0], dtype=np.float32))
d_attract = cuda.to_device(np.array([1], dtype=np.float32))  # Start with attraction enabled

# Thread-safe queue for communication between simulation and web server
data_queue = queue.Queue(maxsize=10)
simulation_running = True
connected_clients = set()

def process_canvas_forces(image):
    """Process canvas image to create a force map based on pixel colors"""
    try:
        # Convert to numpy array for analysis
        import numpy as np
        img_array = np.array(image)
        
        # Convert to grayscale for simpler processing
        if len(img_array.shape) == 3:
            # RGB to grayscale conversion
            gray_array = np.mean(img_array, axis=2).astype(np.uint8)
        else:
            gray_array = img_array
        
        # Save debug image
        Image.fromarray(gray_array).save('gray_array.png')
        
        # Create force map: 0 for white (no force), 1 for black (apply force)
        # Assuming white background (255) and black drawing (0)
        # Threshold at 128 - anything darker than 128 becomes a force pixel
        force_map = (gray_array < 128).astype(np.float32)
        
        # Save debug force map
        debug_force = (force_map * 255).astype(np.uint8)
        Image.fromarray(debug_force).save('force_map_debug.png')
        
        print(f"Force map created: {np.sum(force_map)} force pixels out of {force_map.size} total pixels")
        return force_map
        
    except Exception as e:
        print(f"Error processing canvas for forces: {e}")
        import traceback
        traceback.print_exc()
        return None

def render_force_map_debug():
    """Render force maps as a debug image for visualization"""
    try:
        # Create a white background
        img = Image.new('RGB', (WIDTH, HEIGHT), 'white')
        img_array = np.array(img)
        
        # Add Player 1 force map (bright red)
        if player1_force_map is not None:
            red_overlay = np.zeros_like(img_array)
            red_overlay[:, :, 0] = (player1_force_map).astype(np.uint8)
            red_overlay[:, :, 1] = 0
            red_overlay[:, :, 2] = 0
            img_array = np.clip(img_array + red_overlay, 0, 255)
        
        # Add Player 2 force map (darker red)
        if player2_force_map is not None:
            red_overlay2 = np.zeros_like(img_array)
            red_overlay2[:, :, 0] = (player2_force_map).astype(np.uint8)
            red_overlay2[:, :, 1] = 0
            red_overlay2[:, :, 2] = 0
            img_array = np.clip(img_array + red_overlay2, 0, 255)
        
        # Convert back to PIL Image
        debug_img = Image.fromarray(img_array)
        
        # Save debug image
        debug_img.save('force_map_debug.png')
        print("Force map debug image saved as 'force_map_debug.png'")
        
    except Exception as e:
        print(f"Error creating force map debug image: {e}")

def render_simulation_image(positions):
    """Render the simulation as an image with force maps as red overlay"""
    try:
        # Create a white background image
        img = Image.new('RGB', (WIDTH, HEIGHT), 'white')
        draw = ImageDraw.Draw(img)
        
        # Draw particles as blue circles
        particle_count = 0
        for pos in positions:
            x, y = int(pos[0]), int(pos[1])
            if 0 <= x < WIDTH and 0 <= y < HEIGHT:
                # Draw a small blue circle for each particle
                draw.ellipse([x-2, y-2, x+2, y+2], fill='blue')
                particle_count += 1
        
        print(f"Rendered {particle_count} particles")
        
        # Convert to numpy array for force map overlay
        import numpy as np
        img_array = np.array(img)
        
        # Add force maps as red overlay
        if player1_force_map is not None:
            # Player 1 force map as red overlay (semi-transparent)
            red_overlay = np.zeros_like(img_array)
            red_overlay[:, :, 0] = (player1_force_map * 255).astype(np.uint8)  # Red channel
            red_overlay[:, :, 1] = 0  # Green channel
            red_overlay[:, :, 2] = 0  # Blue channel
            
            # Blend with original image (50% opacity)
            img_array = np.clip(img_array + (red_overlay * 0.5).astype(np.uint8), 0, 255)
            
            # Debug info
            force_pixels = np.sum(player1_force_map)
            print(f"Player 1 force map: {force_pixels} force pixels ({force_pixels/(WIDTH*HEIGHT)*100:.2f}% of canvas)")
        
        if player2_force_map is not None:
            # Player 2 force map as darker red overlay
            red_overlay2 = np.zeros_like(img_array)
            red_overlay2[:, :, 0] = (player2_force_map * 200).astype(np.uint8)  # Darker red
            red_overlay2[:, :, 1] = 0
            red_overlay2[:, :, 2] = 0
            
            # Blend with current image (30% opacity)
            img_array = np.clip(img_array + (red_overlay2 * 0.3).astype(np.uint8), 0, 255)
            
            # Debug info
            force_pixels = np.sum(player2_force_map)
            print(f"Player 2 force map: {force_pixels} force pixels ({force_pixels/(WIDTH*HEIGHT)*100:.2f}% of canvas)")
        
        # Convert back to PIL Image
        img_with_forces = Image.fromarray(img_array)
        
        # Convert to base64
        import io
        buffer = io.BytesIO()
        img_with_forces.save(buffer, format='PNG')
        img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        print(f"Image encoded with force maps, length: {len(img_base64)}")
        return img_base64
    except Exception as e:
        print(f"Error rendering image: {e}")
        import traceback
        traceback.print_exc()
        # Return a simple test image
        img = Image.new('RGB', (WIDTH, HEIGHT), 'red')
        draw = ImageDraw.Draw(img)
        draw.rectangle([10, 10, 100, 100], fill='blue')
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        return base64.b64encode(buffer.getvalue()).decode('utf-8')

@cuda.jit
def update(pos, vel, dt, gravity, damping, radius, width, height, target, attract, force_map1, force_map2, pull_strength):
    i = cuda.grid(1)
    if i >= pos.shape[0]:
        return

    # Gravity pulls down (y increases downward)
    vel[i, 1] += gravity * dt

    # If attract flag set, apply pulling force based on force maps
    if attract[0] == 1:
        # Get particle position as pixel coordinates
        pixel_x = int(pos[i, 0])
        pixel_y = int(pos[i, 1])
        
        # Check bounds
        if 0 <= pixel_x < width and 0 <= pixel_y < height:
            # Check force map 1 (player 1) - if pixel is black, apply force
            if force_map1[pixel_y, pixel_x] > 0:
                vel[i, 0] += pull_strength[0] * dt
                vel[i, 1] += pull_strength[1] * dt
            
            # Check force map 2 (player 2) - if pixel is black, apply force
            if force_map2[pixel_y, pixel_x] > 0:
                vel[i, 0] += pull_strength[0] * dt
                vel[i, 1] += pull_strength[1] * dt

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
    global positions, velocities, player1_force_map, player2_force_map
    blocks = (NUM_PARTICLES + TPB - 1) // TPB
    
    # Initialize empty force maps
    if player1_force_map is None:
        player1_force_map = np.zeros((HEIGHT, WIDTH), dtype=np.float32)
    if player2_force_map is None:
        player2_force_map = np.zeros((HEIGHT, WIDTH), dtype=np.float32)
    
    # Create CUDA device arrays for force maps
    d_force_map1 = cuda.to_device(player1_force_map)
    d_force_map2 = cuda.to_device(player2_force_map)
    
    while simulation_running:
        start_time = time.time()
        
        # Update force maps if they've changed
        if player1_force_map is not None:
            d_force_map1 = cuda.to_device(player1_force_map)
        if player2_force_map is not None:
            d_force_map2 = cuda.to_device(player2_force_map)
        
        # Run CUDA kernel with force maps
        update[blocks, TPB](d_pos, d_vel, DT, GRAVITY, DAMPING, RADIUS, WIDTH, HEIGHT, 
                           d_target, d_attract, d_force_map1, d_force_map2, RECT_FORCE)
        cuda.synchronize()
        
        # Copy data back to host
        positions = d_pos.copy_to_host()
        
        # Put data in queue for web server
        try:
            data_queue.put_nowait({
                'positions': positions.copy().tolist(),
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
    dcc.Store(id='rect-store', data={'x': WIDTH//2 - rect_width//2, 'y': HEIGHT//2 - rect_height//2}),
    
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
    global simulation_running, positions, velocities, player1_rectangle, player2_rectangle
    
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
    
    # Get latest data from simulation thread
    try:
        data = data_queue.get_nowait()
        positions = data['positions']
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
    
    # Add player rectangles (for Dash admin view)
    if player1_rectangle:
        fig.add_trace(go.Scatter(
            x=[player1_rectangle[0], player1_rectangle[0] + player1_rectangle[2], player1_rectangle[0] + player1_rectangle[2], player1_rectangle[0], player1_rectangle[0]],
            y=[player1_rectangle[1], player1_rectangle[1], player1_rectangle[1] + player1_rectangle[3], player1_rectangle[1] + player1_rectangle[3], player1_rectangle[1]],
            mode='lines',
            fill='toself',
            fillcolor='rgba(255, 0, 0, 0.5)',
            line=dict(color='red', width=1),
            name='Player 1',
            showlegend=False
        ))
    
    if player2_rectangle:
        fig.add_trace(go.Scatter(
            x=[player2_rectangle[0], player2_rectangle[0] + player2_rectangle[2], player2_rectangle[0] + player2_rectangle[2], player2_rectangle[0], player2_rectangle[0]],
            y=[player2_rectangle[1], player2_rectangle[1], player2_rectangle[1] + player2_rectangle[3], player2_rectangle[1] + player2_rectangle[3], player2_rectangle[1]],
            mode='lines',
            fill='toself',
            fillcolor='rgba(0, 0, 255, 0.5)',
            line=dict(color='blue', width=1),
            name='Player 2',
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
    global client_rectangles
    
    if not current_rect:
        current_rect = {'x': WIDTH//2 - rect_width//2, 'y': HEIGHT//2 - rect_height//2}
    
    # Handle click events - create a default rectangle for Dash users
    if click_data:
        point = click_data['points'][0]
        x, y = point['x'], point['y']
        # Move rectangle to clicked position (center it)
        current_rect['x'] = x - rect_width // 2
        current_rect['y'] = y - rect_height // 2
        
        # Update the default rectangle for Dash users
        if 'dash_default' not in client_rectangles:
            client_rectangles['dash_default'] = [rect_width, rect_height, rect_width, rect_height]
        client_rectangles['dash_default'][0] = current_rect['x']
        client_rectangles['dash_default'][1] = current_rect['y']
    
    return current_rect

# Add Flask route for external updates
@server.route('/update_rect', methods=['POST'])
def update_rect():
    global client_rectangles
    data = request.get_json()
    x = data.get('x', WIDTH//2)
    y = data.get('y', HEIGHT//2)
    client_id = data.get('client_id', 'external')
    
    # Update rectangle position for external client
    if client_id not in client_rectangles:
        client_rectangles[client_id] = [rect_width, rect_height, rect_width, rect_height]
    client_rectangles[client_id][0] = x
    client_rectangles[client_id][1] = y
    
    return jsonify({'status': 'success'})

# Add Flask route for force map debug
@server.route('/debug_force_map', methods=['GET'])
def debug_force_map():
    render_force_map_debug()
    return jsonify({
        'status': 'success',
        'player1_force_pixels': int(np.sum(player1_force_map)) if player1_force_map is not None else 0,
        'player2_force_pixels': int(np.sum(player2_force_map)) if player2_force_map is not None else 0,
        'total_pixels': WIDTH * HEIGHT
    })

# SocketIO event handlers for React app
@socketio.on('connect')
def handle_connect():
    global player1_id, player2_id, player1_force_map, player2_force_map
    print(f'React client connected: {request.sid}')
    connected_clients.add(request.sid)
    
    # Assign player slot
    if player1_id is None:
        player1_id = request.sid
        player1_force_map = np.zeros((HEIGHT, WIDTH), dtype=np.float32)
        player_number = 1
    elif player2_id is None:
        player2_id = request.sid
        player2_force_map = np.zeros((HEIGHT, WIDTH), dtype=np.float32)
        player_number = 2
    else:
        # Max players reached
        emit('error', {'message': 'Maximum players reached (2)'})
        return
    
    emit('connected', {
        'status': 'connected', 
        'client_id': request.sid,
        'player_number': player_number,
        'max_players': 2
    })

@socketio.on('disconnect')
def handle_disconnect():
    global player1_id, player2_id, player1_force_map, player2_force_map
    print(f'React client disconnected: {request.sid}')
    connected_clients.discard(request.sid)
    
    # Remove this client's force map
    if request.sid == player1_id:
        player1_id = None
        player1_force_map = np.zeros((HEIGHT, WIDTH), dtype=np.float32)
    elif request.sid == player2_id:
        player2_id = None
        player2_force_map = np.zeros((HEIGHT, WIDTH), dtype=np.float32)

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

@socketio.on('update_canvas')
def handle_update_canvas(data):
    global player1_force_map, player2_force_map
    print(f"Received canvas data from {request.sid}")
    print(f"Data keys: {list(data.keys())}")
    print(f"Canvas data length: {len(data.get('canvas_data', ''))}")
    
    try:
        # Extract canvas data
        canvas_data = data.get('canvas_data')
        if not canvas_data:
            print("No canvas data received")
            return
        
        # Remove the data URL prefix to get just the base64 data
        if canvas_data.startswith('data:image/png;base64,'):
            canvas_data = canvas_data.split(',')[1]
        
        # Decode base64 image
        import io
        image_bytes = base64.b64decode(canvas_data)
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Save received image for debugging
        image.save('canvas_data.png')
        print(f"Saved received canvas as 'canvas_data.png'")
        
        # Process canvas to create force map
        force_map = process_canvas_forces(image)
        
        if force_map is not None:
            # Update the appropriate player's force map
            if request.sid == player1_id:
                player1_force_map = force_map
                print(f"Updated Player 1 force map: {np.sum(force_map)} force pixels")
            elif request.sid == player2_id:
                player2_force_map = force_map
                print(f"Updated Player 2 force map: {np.sum(force_map)} force pixels")
            else:
                print(f"Unknown client ID: {request.sid}")
        else:
            print("Failed to process canvas for force map")
            
    except Exception as e:
        print(f"Error processing canvas data: {e}")
        import traceback
        traceback.print_exc()

# Note: Old rectangle handler removed - now using force maps based on canvas pixels

def broadcast_simulation_data():
    """Broadcast simulation image to all connected React clients"""
    while True:
        try:
            data = data_queue.get(timeout=1)
            if connected_clients:
                # Render the simulation as an image (particles only)
                image_base64 = render_simulation_image(data['positions'])
                
                # Send data to each client
                for client_id in connected_clients:
                    personalized_data = {
                        'image': image_base64,
                        'timestamp': data['timestamp']
                    }
                    
                    socketio.emit('simulation_image', personalized_data, room=client_id)
        except queue.Empty:
            continue
        except Exception as e:
            print(f"Error broadcasting data: {e}")
            import traceback
            traceback.print_exc()

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