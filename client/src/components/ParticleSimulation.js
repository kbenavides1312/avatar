import React, { useRef, useEffect, useState, useCallback } from 'react';
import io from 'socket.io-client';
import './ParticleSimulation.css';

const ParticleSimulation = () => {
  const canvasRef = useRef(null);
  const [isConnected, setIsConnected] = useState(false);
  const [simulationRunning, setSimulationRunning] = useState(false);
  const [particles, setParticles] = useState([]);
  const [rectData, setRectData] = useState({ x: 600, y: 200, width: 20, height: 80 });
  const [isDragging, setIsDragging] = useState(false);
  const [dragOffset, setDragOffset] = useState({ x: 0, y: 0 });
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

    socketRef.current.on('simulation_data', (data) => {
      setParticles(data.positions || []);
      setRectData(data.rect_data || { x: 600, y: 200, width: 20, height: 80 });
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

    // Draw rectangle
    ctx.strokeStyle = 'green';
    ctx.lineWidth = 3;
    ctx.strokeRect(rectData.x, rectData.y, rectData.width, rectData.height);

    // Draw connection status
    ctx.fillStyle = isConnected ? 'green' : 'red';
    ctx.fillRect(10, 10, 10, 10);
    ctx.fillStyle = 'black';
    ctx.font = '12px Arial';
    ctx.fillText(isConnected ? 'Connected' : 'Disconnected', 25, 20);
  }, [particles, rectData, isConnected]);

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

  // Mouse event handlers
  const handleMouseDown = useCallback((e) => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    // Check if click is inside rectangle
    if (x >= rectData.x && x <= rectData.x + rectData.width &&
        y >= rectData.y && y <= rectData.y + rectData.height) {
      setIsDragging(true);
      setDragOffset({
        x: x - rectData.x,
        y: y - rectData.y
      });
    }
  }, [rectData]);

  const handleMouseMove = useCallback((e) => {
    if (!isDragging) return;

    const canvas = canvasRef.current;
    if (!canvas) return;

    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    const newX = Math.max(0, Math.min(WIDTH - rectData.width, x - dragOffset.x));
    const newY = Math.max(0, Math.min(HEIGHT - rectData.height, y - dragOffset.y));

    const newRectData = { ...rectData, x: newX, y: newY };
    setRectData(newRectData);

    // Send new position to server
    if (socketRef.current && isConnected) {
      socketRef.current.emit('update_rect', newRectData);
    }
  }, [isDragging, dragOffset, rectData, isConnected]);

  const handleMouseUp = useCallback(() => {
    setIsDragging(false);
  }, []);

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
          onMouseDown={handleMouseDown}
          onMouseMove={handleMouseMove}
          onMouseUp={handleMouseUp}
          onMouseLeave={handleMouseUp}
          className="simulation-canvas"
        />
        <div className="instructions">
          <p>Click and drag the green rectangle to move it</p>
          <p>Particles will be attracted to the rectangle</p>
        </div>
      </div>

      <div className="simulation-info">
        <div className="info-item">
          <strong>Particles:</strong> {particles.length}
        </div>
        <div className="info-item">
          <strong>Status:</strong> {simulationRunning ? 'Running' : 'Stopped'}
        </div>
        <div className="info-item">
          <strong>Rectangle Position:</strong> ({Math.round(rectData.x)}, {Math.round(rectData.y)})
        </div>
      </div>
    </div>
  );
};

export default ParticleSimulation; 