# Particle Simulation React App

This is a React application that displays a real-time particle simulation in the browser. It communicates with a Python server via WebSocket to receive simulation data and send control commands.

## Features

- **Real-time visualization**: Displays particle positions and forces in real-time
- **Interactive controls**: Start, stop, and reset the simulation
- **Drag and drop**: Click and drag the green rectangle to move it
- **Responsive design**: Works on desktop and mobile devices
- **WebSocket communication**: Real-time data streaming from Python server
- **Canvas rendering**: High-performance HTML5 Canvas for smooth animation

## Prerequisites

- Node.js (v14 or higher)
- npm or yarn
- Python server running (see main README)

## Installation

1. Navigate to the client directory:
```bash
cd client
```

2. Install dependencies:
```bash
npm install
```

3. Start the development server:
```bash
npm start
```

The React app will open in your browser at `http://localhost:3000`.

## Usage

1. **Start the Python server first** (in the main project directory):
```bash
python socket_server.py
```

2. **Start the React app** (in the client directory):
```bash
npm start
```

3. **Interact with the simulation**:
   - Click "Start" to begin the simulation
   - Click "Stop" to pause the simulation
   - Click "Reset" to reset all particles
   - Click and drag the green rectangle to move it
   - Watch particles being attracted to the rectangle

## Architecture

```
┌─────────────────┐    WebSocket    ┌─────────────────┐
│   React App     │◄──────────────►│  Python Server  │
│   (Client)      │                 │   (CUDA)        │
└─────────────────┘                 └─────────────────┘
         │                                   │
         ▼                                   ▼
┌─────────────────┐                 ┌─────────────────┐
│  HTML5 Canvas   │                 │  CUDA Kernel    │
│  Rendering      │                 │  Simulation     │
└─────────────────┘                 └─────────────────┘
```

## Components

### ParticleSimulation.js
The main component that handles:
- WebSocket connection to Python server
- Canvas rendering of particles and rectangle
- Mouse interaction for dragging the rectangle
- Control buttons for simulation management
- Real-time data updates

### Key Features

- **Canvas Rendering**: Uses HTML5 Canvas for high-performance graphics
- **WebSocket Communication**: Real-time bidirectional communication
- **Mouse Interaction**: Drag and drop functionality for the force rectangle
- **Responsive Design**: Adapts to different screen sizes
- **Status Indicators**: Shows connection status and simulation state

## WebSocket Events

### Client to Server
- `start_simulation`: Start the particle simulation
- `stop_simulation`: Stop the particle simulation
- `reset_simulation`: Reset all particles to random positions
- `update_rect`: Update rectangle position

### Server to Client
- `connected`: Confirmation of WebSocket connection
- `simulation_data`: Real-time particle and rectangle data
- `simulation_started`: Confirmation that simulation started
- `simulation_stopped`: Confirmation that simulation stopped
- `simulation_reset`: Confirmation that simulation was reset

## Styling

The app uses modern CSS with:
- Gradient backgrounds
- Smooth animations and transitions
- Responsive design for mobile devices
- Glassmorphism effects
- Hover states and visual feedback

## Performance

- **Canvas Optimization**: Efficient rendering with requestAnimationFrame
- **WebSocket Throttling**: Server sends data at ~30 FPS
- **Memory Management**: Proper cleanup of event listeners and animations
- **Responsive Canvas**: Adapts to different screen sizes

## Troubleshooting

1. **Connection Issues**: Make sure the Python server is running on port 8050
2. **Performance Issues**: Reduce particle count in the Python server
3. **Canvas Not Rendering**: Check browser console for errors
4. **WebSocket Errors**: Ensure CORS is properly configured on the server

## Development

To modify the React app:

1. Edit `src/components/ParticleSimulation.js` for main functionality
2. Edit `src/components/ParticleSimulation.css` for styling
3. Edit `src/App.js` to change the main app structure

## Building for Production

```bash
npm run build
```

This creates an optimized production build in the `build` folder.

## Dependencies

- `react`: UI framework
- `react-dom`: DOM rendering
- `socket.io-client`: WebSocket communication
- `react-scripts`: Development and build tools 