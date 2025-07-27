# Particle Simulation - Full Stack Application

This project consists of a Python backend with CUDA-accelerated particle simulation and a React frontend for real-time visualization.

## Project Structure

```
avatar/
├── socket_server.py          # Python WebSocket server with CUDA simulation
├── test.py                   # Original matplotlib simulation
├── requirements.txt          # Python dependencies
├── client/                   # React frontend
│   ├── src/
│   │   ├── components/
│   │   │   ├── ParticleSimulation.js
│   │   │   └── ParticleSimulation.css
│   │   ├── App.js
│   │   └── ...
│   ├── package.json
│   └── README_REACT.md
├── README_DASH.md           # Dash server documentation
└── README_FULL_SETUP.md     # This file
```

## Prerequisites

### Python Backend
- Python 3.7+
- CUDA-compatible GPU
- CUDA toolkit installed
- pip package manager

### React Frontend
- Node.js 14+
- npm or yarn

## Quick Start

### 1. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 2. Install React Dependencies

```bash
cd client
npm install
cd ..
```

### 3. Start the Python Server

```bash
python socket_server.py
```

The server will start on `http://localhost:8050` and display:
```
Starting Socket.IO server...
Make sure the React app is running on http://localhost:3000
```

### 4. Start the React App

In a new terminal:
```bash
cd client
npm start
```

The React app will open automatically in your browser at `http://localhost:3000`.

## How It Works

### Backend (Python + CUDA)
- **CUDA Simulation**: Runs particle physics on GPU for high performance
- **WebSocket Server**: Uses Flask-SocketIO for real-time communication
- **Threading**: Simulation runs in separate thread, independent of web server
- **Data Streaming**: Sends particle positions and rectangle data at ~30 FPS

### Frontend (React)
- **Canvas Rendering**: HTML5 Canvas for smooth particle visualization
- **WebSocket Client**: Real-time communication with Python server
- **Interactive Controls**: Start/stop/reset buttons and drag-and-drop rectangle
- **Responsive Design**: Works on desktop and mobile devices

## Features

### Simulation Features
- **3000 particles** with realistic physics
- **Gravity and damping** effects
- **Particle collisions** with viscosity
- **Force rectangle** that attracts particles
- **Real-time interaction** via drag and drop

### UI Features
- **Modern design** with gradient backgrounds
- **Real-time status** indicators
- **Responsive layout** for all screen sizes
- **Smooth animations** and transitions
- **Interactive controls** with visual feedback

## Usage

1. **Start the simulation**: Click the "Start" button
2. **Move the force rectangle**: Click and drag the green rectangle
3. **Stop the simulation**: Click the "Stop" button
4. **Reset particles**: Click the "Reset" button to randomize positions

## Performance

- **Backend**: CUDA acceleration for 3000+ particles at 30 FPS
- **Frontend**: Canvas rendering with requestAnimationFrame
- **Network**: WebSocket for low-latency real-time communication
- **Memory**: Efficient data structures and cleanup

## Troubleshooting

### Common Issues

1. **CUDA Errors**
   ```
   Solution: Ensure CUDA is properly installed and GPU is compatible
   ```

2. **Port Conflicts**
   ```
   Python server: Change port in socket_server.py
   React app: Change port in client/package.json
   ```

3. **Connection Issues**
   ```
   Check that both servers are running:
   - Python: http://localhost:8050
   - React: http://localhost:3000
   ```

4. **Performance Issues**
   ```
   Reduce NUM_PARTICLES in socket_server.py
   Increase frame rate limits
   ```

### Debug Mode

Enable debug logging:
```python
# In socket_server.py
socketio.run(app, debug=True, host='0.0.0.0', port=8050)
```

## Development

### Modifying the Simulation
- Edit `socket_server.py` for physics parameters
- Adjust `NUM_PARTICLES`, `GRAVITY`, `PULL_STRENGTH`, etc.
- Modify CUDA kernel in the `update` function

### Modifying the UI
- Edit `client/src/components/ParticleSimulation.js` for functionality
- Edit `client/src/components/ParticleSimulation.css` for styling
- Add new components in `client/src/components/`

### Adding Features
- New WebSocket events in `socket_server.py`
- New UI components in React
- Additional simulation parameters

## Production Deployment

### Backend
```bash
# Install production dependencies
pip install gunicorn

# Run with gunicorn
gunicorn -k gevent -w 1 --bind 0.0.0.0:8050 socket_server:app
```

### Frontend
```bash
cd client
npm run build
# Serve build/ directory with nginx or similar
```

## API Reference

### WebSocket Events

#### Client → Server
- `start_simulation`: Start particle simulation
- `stop_simulation`: Stop particle simulation  
- `reset_simulation`: Reset particle positions
- `update_rect`: Update rectangle position `{x, y}`

#### Server → Client
- `connected`: Connection confirmation
- `simulation_data`: Particle data `{positions, rect_data, timestamp}`
- `simulation_started`: Start confirmation
- `simulation_stopped`: Stop confirmation
- `simulation_reset`: Reset confirmation

### Data Format
```javascript
// Particle positions
positions: [[x1, y1], [x2, y2], ...]

// Rectangle data
rect_data: {x: number, y: number, width: number, height: number}
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test both backend and frontend
5. Submit a pull request

## License

This project is open source and available under the MIT License. 