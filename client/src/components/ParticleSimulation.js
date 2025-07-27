import React, { useRef, useEffect, useState, useCallback } from 'react';
import io from 'socket.io-client';
import './ParticleSimulation.css';

const ParticleSimulation = () => {
  const canvasRef = useRef(null);
  const [isConnected, setIsConnected] = useState(false);
  const [simulationRunning, setSimulationRunning] = useState(false);
  const [simulationImage, setSimulationImage] = useState(null);
  const [myRectangle, setMyRectangle] = useState(null);
  const [myClientId, setMyClientId] = useState(null);
  const [playerNumber, setPlayerNumber] = useState(null);
  const socketRef = useRef(null);
  const animationFrameRef = useRef(null);
  const imageCacheRef = useRef(null);

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
      setPlayerNumber(data.player_number);
      console.log('Connected as Player', data.player_number);
    });

    socketRef.current.on('error', (data) => {
      console.error('Server error:', data.message);
      alert(data.message);
    });

    socketRef.current.on('simulation_image', (data) => {
      // Pre-load image for better performance
      if (data.image) {
        const img = new Image();
        img.onload = () => {
          imageCacheRef.current = img;
          setSimulationImage(data.image);
        };
        img.src = `data:image/png;base64,${data.image}`;
      }
      
      if (data.my_rectangle) {
        setMyRectangle(data.my_rectangle);
      }
    });

    return () => {
      if (socketRef.current) {
        socketRef.current.disconnect();
      }
    };
  }, []);

  // Canvas drawing function - optimized for video streaming
  const drawCanvas = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    
    // Draw server-rendered image if available
    if (imageCacheRef.current) {
      // Clear and draw new image
      ctx.clearRect(0, 0, WIDTH, HEIGHT);
      ctx.drawImage(imageCacheRef.current, 0, 0, WIDTH, HEIGHT);
      
      // Overlay my rectangle on top
      if (myRectangle) {
        ctx.fillStyle = 'rgba(0, 0, 0, 0.5)';
        ctx.fillRect(myRectangle[0], myRectangle[1], myRectangle[2], myRectangle[3]);
      }
      
      // Draw connection status
      ctx.fillStyle = isConnected ? 'green' : 'red';
      ctx.beginPath();
      ctx.arc(10, 10, 5, 0, 2 * Math.PI);
      ctx.fill();
      
      ctx.fillStyle = 'black';
      ctx.font = '12px Arial';
      ctx.fillText(isConnected ? 'Connected' : 'Disconnected', 25, 20);
    } else {
      // Fallback: show loading or disconnected state
      ctx.clearRect(0, 0, WIDTH, HEIGHT);
      ctx.fillStyle = 'lightgray';
      ctx.fillRect(0, 0, WIDTH, HEIGHT);
      
      ctx.fillStyle = 'black';
      ctx.font = '16px Arial';
      ctx.textAlign = 'center';
      ctx.fillText('Waiting for simulation...', WIDTH/2, HEIGHT/2);
      
      // Draw connection status
      ctx.fillStyle = isConnected ? 'green' : 'red';
      ctx.beginPath();
      ctx.arc(10, 10, 5, 0, 2 * Math.PI);
      ctx.fill();
      
      ctx.fillStyle = 'black';
      ctx.font = '12px Arial';
      ctx.textAlign = 'left';
      ctx.fillText(isConnected ? 'Connected' : 'Disconnected', 25, 20);
    }
  }, [simulationImage, myRectangle, isConnected]);

  // Animation loop - optimized for video streaming
  useEffect(() => {
    let lastDrawTime = 0;
    const targetFPS = 60;
    const frameInterval = 1000 / targetFPS;
    
    const animate = (currentTime) => {
      // Throttle drawing to target FPS
      if (currentTime - lastDrawTime >= frameInterval) {
        drawCanvas();
        lastDrawTime = currentTime;
      }
      animationFrameRef.current = requestAnimationFrame(animate);
    };
    
    if (simulationRunning) {
      animationFrameRef.current = requestAnimationFrame(animate);
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
    if (!canvas || !myRectangle) return;

    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    // Move my rectangle to clicked position (center it on the click)
    const newX = Math.max(0, Math.min(WIDTH - myRectangle[2], x - myRectangle[2] / 2));
    const newY = Math.max(0, Math.min(HEIGHT - myRectangle[3], y - myRectangle[3] / 2));

    // Send new position to server
    if (socketRef.current && isConnected) {
      socketRef.current.emit('update_rect', { x: newX, y: newY });
    }
  }, [myRectangle, isConnected]);

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
        <h1>Particle Simulation - 2 Players</h1>
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
          {playerNumber && (
            <span className="player-indicator">
              Player {playerNumber}
            </span>
          )}
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
          <strong>Player:</strong> {playerNumber ? `Player ${playerNumber}` : 'Waiting...'}
        </div>
        <div className="info-item">
          <strong>Status:</strong> {simulationRunning ? 'Running' : 'Stopped'}
        </div>
        <div className="info-item">
          <strong>Image Status:</strong> {simulationImage ? 'Receiving' : 'Waiting...'}
        </div>
        <div className="info-item">
          <strong>My Rectangle:</strong> {myRectangle ? 'Active' : 'Waiting...'}
        </div>
      </div>
    </div>
  );
};

export default ParticleSimulation; 