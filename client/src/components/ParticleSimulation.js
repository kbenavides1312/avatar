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
  const [isDrawing, setIsDrawing] = useState(false);
  const [showDrawing, setShowDrawing] = useState(true);
  const socketRef = useRef(null);
  const animationFrameRef = useRef(null);
  const imageCacheRef = useRef(null);
  const drawingCanvasRef = useRef(null);

  // Canvas dimensions
  const WIDTH = 1200;
  const HEIGHT = 400;

  // Function to capture canvas data as base64
  const captureCanvasData = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return null;
    
    return canvas.toDataURL('image/png');
  }, []);

  // Function to capture drawing canvas data as base64
  const captureDrawingData = useCallback(() => {
    const canvas = drawingCanvasRef.current;
    if (!canvas) {
      console.log('No drawing canvas found');
      return null;
    }
    
    // Get the canvas context and image data
    const ctx = canvas.getContext('2d');
    const imageData = ctx.getImageData(0, 0, WIDTH, HEIGHT);
    const data = imageData.data;
    
    // Create a copy of the original data
    const originalData = new Uint8ClampedArray(data);
    
    // Replace transparent pixels with white, and convert light gray to black for forces
    for (let i = 0; i < data.length; i += 4) {
      // If pixel is transparent (alpha < 255), make it white
      if (data[i + 3] < 10) {
        data[i] = 255;     // Red = 255 (white)
        data[i + 1] = 255; // Green = 255 (white)
        data[i + 2] = 255; // Blue = 255 (white)
        data[i + 3] = 255; // Alpha = 255 (opaque)
      }
      // If pixel is dark gray (for drawing), convert to black for forces
      else if (data[i] === 51 && data[i + 1] === 51 && data[i + 2] === 51) {
        data[i] = 0;       // Red = 0 (black)
        data[i + 1] = 0;   // Green = 0 (black)
        data[i + 2] = 0;   // Blue = 0 (black)
        data[i + 3] = 255; // Alpha = 255 (opaque)
      }
    }
    
    // Put the modified data back to canvas temporarily
    ctx.putImageData(imageData, 0, 0);
    
    // Capture the canvas with white background
    const dataURL = canvas.toDataURL('image/png');
    console.log('Captured drawing canvas with white background, data length:', dataURL.length);
    
    // Restore the original canvas data
    const originalImageData = new ImageData(originalData, WIDTH, HEIGHT);
    ctx.putImageData(originalImageData, 0, 0);
    
    return dataURL;
  }, []);

  // Function to send current canvas state to server
  const sendCanvasToServer = useCallback(() => {
    if (socketRef.current && isConnected) {
      const canvasData = captureDrawingData();
      if (canvasData) {
        socketRef.current.emit('update_canvas', { 
          canvas_data: canvasData
        });
      }
    }
  }, [captureDrawingData, isConnected]);

  // Initialize drawing canvas
  useEffect(() => {
    const canvas = drawingCanvasRef.current;
    if (canvas) {
      const ctx = canvas.getContext('2d');
      // Clear canvas and make it transparent
      ctx.clearRect(0, 0, WIDTH, HEIGHT);
      console.log('Initialized drawing canvas');
    }
  }, []);

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
          // Clean up old image to prevent memory leak
          if (imageCacheRef.current) {
            imageCacheRef.current.onload = null;
            imageCacheRef.current.onerror = null;
          }
          imageCacheRef.current = img;
          setSimulationImage(data.image);
        };
        img.onerror = () => {
          console.error('Failed to load simulation image');
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
      // Clean up image cache to prevent memory leak
      if (imageCacheRef.current) {
        imageCacheRef.current.onload = null;
        imageCacheRef.current.onerror = null;
        imageCacheRef.current = null;
      }
    };
  }, []);

  // Canvas drawing function - optimized for video streaming
  const drawCanvas = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    
    // Clear the canvas
    ctx.clearRect(0, 0, WIDTH, HEIGHT);
    
    // Draw server-rendered simulation image if available
    if (imageCacheRef.current) {
      ctx.drawImage(imageCacheRef.current, 0, 0, WIDTH, HEIGHT);
    } else {
      // Fallback: show loading or disconnected state
      ctx.fillStyle = 'lightgray';
      ctx.fillRect(0, 0, WIDTH, HEIGHT);
      ctx.globalAlpha = 0.5;
      ctx.fillStyle = 'black';
      ctx.font = '16px Arial';
      ctx.textAlign = 'center';
      ctx.fillText('Waiting for simulation...', WIDTH/2, HEIGHT/2);
    }
    
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
    
    if (!imageCacheRef.current) {
      ctx.fillStyle = 'black';
      ctx.globalAlpha = 0.5;
      ctx.font = '12px Arial';
      ctx.textAlign = 'left';
      ctx.fillText(isConnected ? 'Connected' : 'Disconnected', 25, 20);
    }
  }, [simulationImage, myRectangle, isConnected]);

  // Animation loop - optimized for video streaming
  useEffect(() => {
    let lastDrawTime = 0;
    let lastSendTime = 0;
    const targetFPS = 60;
    const frameInterval = 1000 / targetFPS;
    const sendInterval = 1000; // Send canvas data every second
    
    const animate = (currentTime) => {
      // Throttle drawing to target FPS
      if (currentTime - lastDrawTime >= frameInterval) {
        drawCanvas();
        lastDrawTime = currentTime;
      }
      
      // Periodically send canvas data to server
      if (currentTime - lastSendTime >= sendInterval) {
        sendCanvasToServer();
        lastSendTime = currentTime;
      }
      
      animationFrameRef.current = requestAnimationFrame(animate);
    };
    
    if (simulationRunning) {
      animationFrameRef.current = requestAnimationFrame(animate);
    }

    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
        animationFrameRef.current = null;
      }
    };
  }, [drawCanvas, simulationRunning, sendCanvasToServer]);

  // Drawing event handlers
  const getCanvasCoordinates = useCallback((e) => {
    const canvas = drawingCanvasRef.current;
    if (!canvas) return { x: 0, y: 0 };
    
    const rect = canvas.getBoundingClientRect();
    return {
      x: e.clientX - rect.left,
      y: e.clientY - rect.top
    };
  }, []);

  const handleMouseDown = useCallback((e) => {
    if (!showDrawing) return;
    
    setIsDrawing(true);
    
    const canvas = drawingCanvasRef.current;
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    const coords = getCanvasCoordinates(e);
    // Start drawing path
    ctx.beginPath();
    ctx.moveTo(coords.x, coords.y);
    ctx.strokeStyle = 'black'; // Darker gray for better visibility
    ctx.lineWidth = 5;
    ctx.lineCap = 'round';
    
    console.log('Started drawing at:', coords.x, coords.y);
  }, [getCanvasCoordinates, showDrawing]);

  const handleMouseMove = useCallback((e) => {
    if (!isDrawing || !showDrawing) return;
    
    const canvas = drawingCanvasRef.current;
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    const coords = getCanvasCoordinates(e);
    
    // Continue drawing path
    ctx.lineTo(coords.x, coords.y);
    ctx.stroke();
  }, [isDrawing, getCanvasCoordinates, showDrawing]);

  const handleMouseUp = useCallback((e) => {
    setIsDrawing(false);
    
    // Only send if we actually drew something
    if (isDrawing) {
      // Send the updated drawing canvas to server
      const canvasData = captureDrawingData();
      if (canvasData && socketRef.current && isConnected) {
        console.log('Sending drawing canvas to server, data length:', canvasData.length);
        socketRef.current.emit('update_canvas', { 
          canvas_data: canvasData
        });
      } else {
        console.log('No drawing data to send or not connected');
      }
    }
  }, [captureDrawingData, isConnected, isDrawing]);

  // Mouse event handler for clicking to send canvas data (fallback)
  const handleCanvasClick = useCallback((e) => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    // Capture the current canvas state
    const canvasData = captureCanvasData();
    if (!canvasData) return;

    // Send canvas data to server
    if (socketRef.current && isConnected) {
      socketRef.current.emit('update_canvas', { 
        canvas_data: canvasData,
        click_x: e.clientX,
        click_y: e.clientY
      });
    }
  }, [captureCanvasData, isConnected]);

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

  const clearCanvas = () => {
    const canvas = drawingCanvasRef.current;
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, WIDTH, HEIGHT);
    
    // Send cleared canvas to server
    const canvasData = captureDrawingData();
    if (canvasData && socketRef.current && isConnected) {
      socketRef.current.emit('update_canvas', { 
        canvas_data: canvasData
      });
    }
  };

  const testDrawing = () => {
    const canvas = drawingCanvasRef.current;
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    // Clear canvas first
    ctx.clearRect(0, 0, WIDTH, HEIGHT);
    // Draw a darker gray rectangle for visibility
    ctx.fillStyle = '#333333';
    ctx.fillRect(100, 300, 50, 150);
    
    console.log('Created test drawing');
    
    // Send test drawing to server
    const canvasData = captureDrawingData();
    if (canvasData && socketRef.current && isConnected) {
      console.log('Sending test drawing to server');
      socketRef.current.emit('update_canvas', { 
        canvas_data: canvasData
      });
    } else {
      console.log('Failed to capture or send test drawing');
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
          <button 
            onClick={sendCanvasToServer} 
            disabled={!isConnected}
            className="control-btn send-btn"
          >
            Send Canvas
          </button>
          <button 
            onClick={clearCanvas} 
            disabled={!isConnected}
            className="control-btn clear-btn"
          >
            Clear Canvas
          </button>
          <button 
            onClick={testDrawing} 
            disabled={!isConnected}
            className="control-btn test-btn"
          >
            Test Drawing
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
        <div className="canvas-section">
          <h3>Simulation with Overlay Drawing</h3>
          <div className="drawing-controls">
            <button 
              onClick={() => setShowDrawing(!showDrawing)}
              className="control-btn toggle-btn"
            >
              {showDrawing ? 'Hide' : 'Show'} Drawing
            </button>
          </div>
          <div className="canvas-stack">
            {/* Background canvas - Simulation */}
            <canvas
              ref={canvasRef}
              width={WIDTH}
              height={HEIGHT}
              onClick={handleCanvasClick}
              className="simulation-canvas"
            />
            
            {/* Foreground canvas - Drawing */}
            <canvas
              ref={drawingCanvasRef}
              width={WIDTH}
              height={HEIGHT}
              onMouseDown={handleMouseDown}
              onMouseMove={handleMouseMove}
              onMouseUp={handleMouseUp}
              onMouseLeave={handleMouseUp}
              className="drawing-canvas"
              style={{ 
                cursor: isDrawing ? 'crosshair' : 'default',
                display: showDrawing ? 'block' : 'none'
              }}
            />
          </div>
        </div>
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
          <strong>Canvas Mode:</strong> Overlay - Drawing Canvas Over Simulation
        </div>
        <div className="info-item">
          <strong>Instructions:</strong> Draw on top layer - black pixels apply pull force
        </div>
        <div className="info-item">
          <strong>Drawing:</strong> {isDrawing ? 'Active' : 'Inactive'}
        </div>
        <div className="info-item">
          <strong>Drawing Canvas:</strong> {showDrawing ? 'Visible' : 'Hidden'}
        </div>
      </div>
    </div>
  );
};

export default ParticleSimulation; 