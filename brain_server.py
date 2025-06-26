"""
Main brain server - coordinates robot connections and neural processing
"""
import asyncio
import time
import websockets
import msgpack
from typing import Dict, Set
from neural_interface import EnhancedNeuralInterface
from robot_connection import RobotConnection
import config

class BrainServer:
    """
    Main brain server that coordinates everything
    """
    
    def __init__(self):
        self.neural_interface = EnhancedNeuralInterface(device=config.NEURAL_DEVICE)
        self.robot_connections: Dict[str, RobotConnection] = {}
        self.connection_counter = 0
        
        self.server_start_time = time.time()
        self.is_running = False
        
        print("🧠 Brain server initialized")
    
    async def start_server(self):
        """
        Start the brain server
        """
        print(f"🧠 Starting brain server on {config.SERVER_HOST}:{config.SERVER_PORT}")
        
        # Start background monitoring
        asyncio.create_task(self._monitor_connections())
        asyncio.create_task(self._monitor_neural_interface())
        asyncio.create_task(self._monitor_learning_progress())
        
        self.is_running = True
        
        # Start WebSocket server with correct handler signature
        async with websockets.serve(
            self._handle_new_connection, 
            config.SERVER_HOST, 
            config.SERVER_PORT
        ):
            print("🧠 Brain server is conscious and ready for robots!")
            await asyncio.Future()  # Run forever
    
    async def _handle_new_connection(self, websocket):
        """
        Handle new robot connection
        Fixed: Added missing 'path' parameter
        """
        if len(self.robot_connections) >= config.MAX_ROBOTS:
            await websocket.close(code=1013, reason="Max robots connected")
            return
        
        # Create robot connection
        self.connection_counter += 1
        robot_id = f"robot_{self.connection_counter:03d}"
        
        robot_conn = RobotConnection(websocket, robot_id, self.neural_interface)
        self.robot_connections[robot_id] = robot_conn
        
        # Handle robot communication
        try:
            await robot_conn.handle_robot_communication()
        finally:
            # Clean up when robot disconnects
            if robot_id in self.robot_connections:
                del self.robot_connections[robot_id]
                print(f"🤖 Robot {robot_id} cleaned up")
    
    async def _monitor_connections(self):
        """
        Monitor robot connections and clean up inactive ones
        """
        while self.is_running:
            try:
                current_time = time.time()
                inactive_robots = []
                
                for robot_id, connection in self.robot_connections.items():
                    if not connection.is_active:
                        inactive_robots.append(robot_id)
                    elif current_time - connection.last_message_time > config.CONNECTION_TIMEOUT:
                        print(f"🤖 Robot {robot_id} timed out")
                        inactive_robots.append(robot_id)
                
                # Remove inactive connections
                for robot_id in inactive_robots:
                    if robot_id in self.robot_connections:
                        del self.robot_connections[robot_id]
                
                await asyncio.sleep(5)  # Check every 5 seconds
                
            except Exception as e:
                print(f"🧠 Connection monitoring error: {e}")
                await asyncio.sleep(5)
    
    async def _monitor_neural_interface(self):
        """
        Monitor neural interface performance
        """
        while self.is_running:
            try:
                status = self.neural_interface.get_status()
                
                # Log status every 30 seconds
                await asyncio.sleep(30)
                print(f"🧠 Neural status: {status['processing_count']} processed, "
                      f"intelligence: {status['intelligence_level']:.3f}")
                
            except Exception as e:
                print(f"🧠 Neural monitoring error: {e}")
                await asyncio.sleep(30)

    async def _monitor_learning_progress(self):
        """Monitor and log learning progress"""
        last_experience_count = 0
        
        while self.is_running:
            try:
                status = self.neural_interface.get_status()
                
                # Check learning progress
                new_experiences = status['total_experiences'] - last_experience_count
                if new_experiences > 0:
                    print(f"🧠 Learning: +{new_experiences} experiences, "
                        f"intelligence: {status['intelligence_level']:.3f}, "
                        f"buffer: {status['buffer_size']}")
                    last_experience_count = status['total_experiences']
                
                await asyncio.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                print(f"🧠 Learning monitoring error: {e}")
                await asyncio.sleep(10)
    
    def get_server_stats(self) -> Dict:
        """
        Get comprehensive server statistics
        """
        uptime = time.time() - self.server_start_time
        
        robot_stats = [conn.get_stats() for conn in self.robot_connections.values()]
        neural_status = self.neural_interface.get_status()
        
        return {
            'server': {
                'uptime_seconds': uptime,
                'is_running': self.is_running,
                'connected_robots': len(self.robot_connections),
                'total_connections': self.connection_counter
            },
            'robots': robot_stats,
            'neural_interface': neural_status
        }