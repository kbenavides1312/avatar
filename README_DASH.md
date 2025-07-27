# Particle Simulation Dash Server

This is a web-based version of the CUDA particle simulation that streams the visualization to a browser using Dash.

## Features

- **Real-time streaming**: The simulation runs on the server and streams updates to the browser
- **Interactive controls**: Start, stop, and reset the simulation
- **Click to move**: Click anywhere on the plot to move the force rectangle
- **CUDA acceleration**: Uses GPU acceleration for particle physics calculations
- **Responsive design**: Works on different screen sizes

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Make sure you have CUDA installed and a compatible GPU.

## Usage

1. Start the Dash server:
```bash
python dash_server.py
```

2. Open your web browser and navigate to:
```
http://127.0.0.1:8050
```

3. The simulation will start automatically. You can:
   - Click anywhere on the plot to move the green force rectangle
   - Use the buttons to start/stop/reset the simulation
   - Watch particles being attracted to the rectangle

## Controls

- **Start Simulation**: Resumes the particle simulation
- **Stop Simulation**: Pauses the particle simulation
- **Reset Particles**: Resets all particles to random positions
- **Click on plot**: Moves the force rectangle to the clicked location

## Technical Details

- **Simulation**: Runs on a separate thread using CUDA for GPU acceleration
- **Streaming**: Uses Dash's interval component to update the plot at ~30 FPS
- **Communication**: Thread-safe queue for data transfer between simulation and web server
- **Performance**: Optimized for web viewing with reduced particle count (3000 particles)

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   CUDA Kernel   │    │  Simulation     │    │   Dash Server   │
│   (GPU)         │◄──►│  Thread         │◄──►│   (Web)         │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                              │                        │
                              ▼                        ▼
                       ┌─────────────────┐    ┌─────────────────┐
                       │  Data Queue     │    │   Browser       │
                       │  (Thread-safe)  │    │   (Client)      │
                       └─────────────────┘    └─────────────────┘
```

## Performance Notes

- The web version uses fewer particles (3000 vs 10000) for better performance
- Frame rate is limited to ~30 FPS for smooth web streaming
- Particle size is reduced for better visual clarity
- The simulation runs continuously on the server regardless of browser state

## Troubleshooting

1. **CUDA errors**: Make sure you have CUDA installed and a compatible GPU
2. **Port conflicts**: If port 8050 is in use, modify the port in the code
3. **Performance issues**: Reduce `NUM_PARTICLES` in the code for better performance
4. **Browser issues**: Try refreshing the page if the simulation doesn't start

## Differences from Original

- Uses Dash/Plotly instead of matplotlib
- Streams to browser instead of local window
- Reduced particle count for web performance
- Simplified mouse interaction
- Added web controls (start/stop/reset) 