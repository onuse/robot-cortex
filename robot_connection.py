"""
Handle individual robot connections
"""
import asyncio
import time
import msgpack
import websockets
from typing import Dict, Any, Optional

class RobotConnection:
    """
    Manages connection to a single robot
    """
    
    def __init__(self, websocket, robot_id: str, neural_interface):
        self.websocket = websocket
        self.robot_id = robot_id
        self.neural_interface = neural_interface
        
        self.connected_at = time.time()
        self.last_message_time = time.time()
        self.message_count = 0
        self.is_active = True
        
        print(f"🤖 Robot {robot_id} connected from {websocket.remote_address}")
    
    async def handle_robot_communication(self):
        """
        Main communication loop for this robot
        """
        try:
            async for raw_message in self.websocket:
                await self._process_robot_message(raw_message)
                
        except websockets.exceptions.ConnectionClosed:
            print(f"🤖 Robot {self.robot_id} disconnected normally")
            self.is_active = False
            
        except websockets.exceptions.ConnectionClosedError:
            print(f"🤖 Robot {self.robot_id} connection closed with error")
            self.is_active = False
            
        except Exception as e:
            print(f"🤖 Error with robot {self.robot_id}: {e}")
            self.is_active = False
    
    async def _process_robot_message(self, raw_message):
        """
        Process a single message from the robot
        """
        try:
            # Decode message
            message = msgpack.unpackb(raw_message)
            self.message_count += 1
            self.last_message_time = time.time()
            
            # Show message received (for debugging)
            if self.message_count % 50 == 1:  # Every 50 messages
                print(f"🤖 Robot {self.robot_id}: {self.message_count} messages processed")
            
            # Validate message structure
            if not self._validate_message(message):
                await self._send_error("Invalid message format")
                return
            
            # Extract sensor data
            sensor_data = message.get('data', {})
            
            # Process through neural interface
            motor_commands = await self.neural_interface.process_sensor_data(sensor_data)
            
            # Send response back to robot
            await self._send_motor_commands(motor_commands)
            
        except msgpack.exceptions.ExtraData as e:
            print(f"🤖 Message decode error for {self.robot_id}: {e}")
            await self._send_error("Message decode error")
            
        except Exception as e:
            print(f"🤖 Message processing error for {self.robot_id}: {e}")
            await self._send_error(f"Processing error: {str(e)}")
    
    def _validate_message(self, message: Dict[str, Any]) -> bool:
        """
        Validate incoming message structure
        """
        required_fields = ['type', 'timestamp', 'data']
        
        if not isinstance(message, dict):
            print(f"🤖 Message not a dict: {type(message)}")
            return False
            
        for field in required_fields:
            if field not in message:
                print(f"🤖 Missing required field: {field}")
                return False
        
        return True
    
    async def _send_motor_commands(self, commands: Dict[str, Any]):
        """
        Send motor commands back to robot
        """
        try:
            response = {
                'type': 'MOTOR_COMMANDS',
                'timestamp': time.time(),
                'robot_id': self.robot_id,
                'data': commands
            }
            
            packed_response = msgpack.packb(response)
            await self.websocket.send(packed_response)
            
        except Exception as e:
            print(f"🤖 Failed to send motor commands to {self.robot_id}: {e}")
            self.is_active = False
    
    async def _send_error(self, error_message: str):
        """
        Send error message to robot
        """
        try:
            response = {
                'type': 'ERROR',
                'timestamp': time.time(),
                'robot_id': self.robot_id,
                'error': error_message
            }
            
            packed_response = msgpack.packb(response)
            await self.websocket.send(packed_response)
            
        except Exception as e:
            print(f"🤖 Failed to send error to {self.robot_id}: {e}")
            self.is_active = False
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get connection statistics
        """
        uptime = time.time() - self.connected_at
        return {
            'robot_id': self.robot_id,
            'uptime_seconds': uptime,
            'message_count': self.message_count,
            'last_message_age': time.time() - self.last_message_time,
            'is_active': self.is_active,
            'messages_per_second': self.message_count / uptime if uptime > 0 else 0
        }