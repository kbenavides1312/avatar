import React, { useRef, useEffect, useState, useCallback } from 'react';
import io from 'socket.io-client';
import './ParticleSimulation.css';

const ParticleSimulation = () => {
  const canvasRef = useRef(null);
  const [isConnected, setIsConnected] = useState(false);
  const [simulationRunning, setSimulationRunning] = useState(false);
  const [particles, setParticles] = useState([]);
  const [clientRectangles, setClientRectangles] = useState({});
  const [myClientId, setMyClientId] = useState(null);
  const socketRef = useRef(null);
  const animationFrameRef = useRef(null);

  // Canvas dimensions
  const WIDTH = 1200;
  const HEIGHT = 400;

  // Initialize WebSocket connection
  useEffect(() => {
    socketRef.current = io('http://localhost:8050', {
      transports: ['websocket', 'polling']
    });

    socketRef.current.on('connect', () => {
      console.log('Connected to server');
      setIsConnected(true);
    });

    socketRef.current.on('disconnect', () => {
      console.log('Disconnected from server');
      setIsConnected(false);
    });

    socketRef.current.on('connected', (data) => {
      setMyClientId(data.client_id);
      console.log('My client ID:', data.client_id);
    });

    socketRef.current.on('simulation_data', (data) => {
      setParticles(data.positions || []);
      setClientRectangles(data.client_rectangles || {});
    });

    return () => {
      if (socketRef.current) {
        socketRef.current.disconnect();
      }
    };
  }, []);

  // Canvas drawing function
  const drawCanvas = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, WIDTH, HEIGHT);

    // Draw particles
    ctx.fillStyle = 'rgba(0, 100, 255, 0.7)';
    particles.forEach(particle => {
      ctx.beginPath();
      ctx.arc(particle[0], particle[1], 2, 0, 2 * Math.PI);
      ctx.fill();
    });

    // Draw all rectangles
    Object.entries(clientRectangles).forEach(([clientId, rectData]) => {
      const isMyRectangle = clientId === myClientId;
      
      if (isMyRectangle) {
        // My rectangle - black with 50% transparency
        ctx.fillStyle = 'rgba(0, 0, 0, 0.5)';
        ctx.fillRect(rectData[0], rectData[1], rectData[2], rectData[3]);
      } else {
        // Other users' rectangles - red with 30% transparency
        ctx.fillStyle = 'rgba(255, 0, 0, 0.3)';
        ctx.fillRect(rectData[0], rectData[1], rectData[2], rectData[3]);
      }
    });

    // Draw connection status
    ctx.fillStyle = isConnected ? 'green' : 'red';
    ctx.fillRect(10, 10, 10, 10);
    ctx.fillStyle = 'black';
    ctx.font = '12px Arial';
    ctx.fillText(isConnected ? 'Connected' : 'Disconnected', 25, 20);
  }, [particles, clientRectangles, myClientId, isConnected]);

  // Animation loop
  useEffect(() => {
    const animate = () => {
      drawCanvas();
      animationFrameRef.current = requestAnimationFrame(animate);
    };
    
    if (simulationRunning) {
      animate();
    }

    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
    };
  }, [drawCanvas, simulationRunning]);

  // Mouse event handler for clicking to move rectangle
  const handleCanvasClick = useCallback((e) => {
    const canvas = canvasRef.current;
    if (!canvas || !myClientId) return;

    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    // Get my rectangle data
    const myRectData = clientRectangles[myClientId];
    if (!myRectData) return;

    // Move my rectangle to clicked position (center it on the click)
    const newX = Math.max(0, Math.min(WIDTH - myRectData[2], x - myRectData[2] / 2));
    const newY = Math.max(0, Math.min(HEIGHT - myRectData[3], y - myRectData[3] / 2));

    // Send new position to server
    if (socketRef.current && isConnected) {
      socketRef.current.emit('update_rect', { x: newX, y: newY });
    }
  }, [clientRectangles, myClientId, isConnected]);

  // Control functions
  const startSimulation = () => {
    if (socketRef.current && isConnected) {
      socketRef.current.emit('start_simulation');
      setSimulationRunning(true);
    }
  };

  const stopSimulation = () => {
    if (socketRef.current && isConnected) {
      socketRef.current.emit('stop_simulation');
      setSimulationRunning(false);
    }
  };

  const resetSimulation = () => {
    if (socketRef.current && isConnected) {
      socketRef.current.emit('reset_simulation');
    }
  };

  return (
    <div className="particle-simulation">
      <div className="simulation-header">
        <h1>Particle Simulation</h1>
        <div className="controls">
          <button 
            onClick={startSimulation} 
            disabled={!isConnected || simulationRunning}
            className="control-btn start-btn"
          >
            Start
          </button>
          <button 
            onClick={stopSimulation} 
            disabled={!isConnected || !simulationRunning}
            className="control-btn stop-btn"
          >
            Stop
          </button>
          <button 
            onClick={resetSimulation} 
            disabled={!isConnected}
            className="control-btn reset-btn"
          >
            Reset
          </button>
        </div>
        <div className="status">
          <span className={`status-indicator ${isConnected ? 'connected' : 'disconnected'}`}>
            {isConnected ? 'Connected' : 'Disconnected'}
          </span>
        </div>
      </div>

      <div className="canvas-container">
        <canvas
          ref={canvasRef}
          width={WIDTH}
          height={HEIGHT}
          onClick={handleCanvasClick}
          className="simulation-canvas"
        />
      </div>

      <div className="simulation-info">
        <div className="info-item">
          <strong>Particles:</strong> {particles.length}
        </div>
        <div className="info-item">
          <strong>Status:</strong> {simulationRunning ? 'Running' : 'Stopped'}
        </div>
        <div className="info-item">
          <strong>Connected Users:</strong> {Object.keys(clientRectangles).length}
        </div>
        <div className="info-item">
          <strong>My Rectangle:</strong> {myClientId ? 'Active' : 'Waiting...'}
        </div>
      </div>
    </div>
  );
};

export default ParticleSimulation; 