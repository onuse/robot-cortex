"""
Enhanced Neural Interface - Now processes real robot experiences
"""
import torch
import torch.nn as nn
import numpy as np
import time
from collections import deque
from dataclasses import dataclass
from typing import Dict, Any, List, Optional

@dataclass
class Experience:
    """Single robot experience for learning"""
    timestamp: float
    sensor_state: torch.Tensor
    action_taken: torch.Tensor  
    outcome_sensors: torch.Tensor
    prediction_error: float
    curiosity_reward: float

class ExperienceBuffer:
    """Store robot experiences for learning"""
    def __init__(self, max_size=10000):
        self.experiences = deque(maxlen=max_size)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    def add_experience(self, experience: Experience):
        self.experiences.append(experience)
    
    def get_recent_batch(self, batch_size=32):
        if len(self.experiences) < batch_size:
            return None
        
        # Get most recent experiences
        recent = list(self.experiences)[-batch_size:]
        return self._experiences_to_tensors(recent)
    
    def _experiences_to_tensors(self, experiences):
        """Convert experience list to GPU tensors"""
        sensor_states = torch.stack([exp.sensor_state for exp in experiences]).to(self.device)
        actions = torch.stack([exp.action_taken for exp in experiences]).to(self.device)
        outcomes = torch.stack([exp.outcome_sensors for exp in experiences]).to(self.device)
        
        return {
            'sensor_states': sensor_states,
            'actions': actions,
            'outcomes': outcomes,
            'prediction_errors': torch.tensor([exp.prediction_error for exp in experiences]).to(self.device),
            'curiosity_rewards': torch.tensor([exp.curiosity_reward for exp in experiences]).to(self.device)
        }

class SimplePredictionNetwork(nn.Module):
    """Basic sensorimotor prediction network"""
    def __init__(self, sensor_dim=8, action_dim=3, hidden_dim=128):
        super().__init__()
        
        # Sensor encoder
        self.sensor_encoder = nn.Sequential(
            nn.Linear(sensor_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim//2)
        )
        
        # Action encoder  
        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, hidden_dim//4),
            nn.ReLU()
        )
        
        # Prediction network
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim//2 + hidden_dim//4, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(), 
            nn.Linear(hidden_dim, sensor_dim)  # Predict next sensor state
        )
        
        # Motor decision network
        self.motor_network = nn.Sequential(
            nn.Linear(hidden_dim//2, hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, action_dim),
            nn.Tanh()  # Output between -1 and 1
        )
    
    def forward(self, sensor_input, action_input=None):
        sensor_features = self.sensor_encoder(sensor_input)
        
        if action_input is not None:
            # Prediction mode: given sensors + action, predict outcome
            action_features = self.action_encoder(action_input)
            combined = torch.cat([sensor_features, action_features], dim=-1)
            predicted_sensors = self.predictor(combined)
            return predicted_sensors
        else:
            # Decision mode: given sensors, decide action
            motor_output = self.motor_network(sensor_features)
            return motor_output

class EnhancedNeuralInterface:
    """Enhanced neural interface that learns from robot experiences"""
    
    def __init__(self, device='cuda'):
        self.device = device
        print(f"🧠 Enhanced neural interface initializing on {device}")
        
        # Initialize neural networks
        self.prediction_net = SimplePredictionNetwork().to(device)
        self.optimizer = torch.optim.Adam(self.prediction_net.parameters(), lr=0.001)
        
        # Experience storage
        self.experience_buffer = ExperienceBuffer()
        self.previous_sensor_state = None
        self.previous_action = None
        
        # Learning metrics
        self.total_experiences = 0
        self.prediction_accuracy = 0.5  # Start at random
        self.learning_rate = 0.0
        
        print("🧠 Enhanced neural interface ready - learning mode active")
    
    async def process_sensor_data(self, sensor_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process sensor data and return motor commands"""
        
        # Convert sensor data to tensor
        sensor_tensor = self._sensors_to_tensor(sensor_data)
        
        # Learn from previous experience if available
        if self.previous_sensor_state is not None and self.previous_action is not None:
            await self._learn_from_experience(sensor_tensor)
        
        # Decide action using neural network
        with torch.no_grad():
            action_tensor = self.prediction_net(sensor_tensor)
        
        # Convert to motor commands
        motor_commands = self._tensor_to_motor_commands(action_tensor, sensor_data)
        
        # Store current state for next learning step
        self.previous_sensor_state = sensor_tensor.clone()
        self.previous_action = action_tensor.clone()
        
        return motor_commands
    
    def _sensors_to_tensor(self, sensor_data: Dict[str, Any]) -> torch.Tensor:
        """Convert sensor dictionary to normalized tensor"""
        
        # Extract and normalize key sensor values
        battery = (sensor_data.get('battery_voltage', 12.0) - 10.0) / 4.0  # Normalize 10-14V to 0-1
        distance = min(sensor_data.get('ultrasonic_distance', 100), 100) / 100.0  # Normalize 0-100cm
        power_source = 1.0 if sensor_data.get('power_source') == 'EXTERNAL' else 0.0
        
        # Motor current (if available)
        motor_current = sensor_data.get('motor_current', [0.5, 0.5])
        current_left = min(motor_current[0], 2.0) / 2.0
        current_right = min(motor_current[1], 2.0) / 2.0
        
        # Time-based features
        time_factor = (time.time() % 60) / 60.0  # Cyclical time feature
        
        # Simple visual feature (if camera available)
        visual_complexity = 0.5  # Placeholder - would analyze camera_frame
        
        sensor_vector = torch.tensor([
            battery, distance, power_source, 
            current_left, current_right,
            time_factor, visual_complexity,
            self.total_experiences / 1000.0  # Experience count as feature
        ], dtype=torch.float32).to(self.device)
        
        return sensor_vector
    
    def _tensor_to_motor_commands(self, action_tensor: torch.Tensor, sensor_data: Dict[str, Any]) -> Dict[str, Any]:
        """Convert neural network output to motor commands"""
        
        action = action_tensor.cpu().numpy()
        
        # Convert neural output [-1,1] to PWM values [1400,1600]
        motor_left = int(1500 + action[0] * 100)   # Left motor
        motor_right = int(1500 + action[1] * 100)  # Right motor
        servo_camera = int(1500 + action[2] * 200) # Camera servo
        
        # Clamp to safe ranges
        motor_left = max(1400, min(1600, motor_left))
        motor_right = max(1400, min(1600, motor_right))
        servo_camera = max(1300, min(1700, servo_camera))
        
        # Add curiosity bonus for exploration
        curiosity_level = self._calculate_curiosity(sensor_data)
        
        return {
            'motor_left_pwm': motor_left,
            'motor_right_pwm': motor_right,
            'servo_camera_pwm': servo_camera,
            'buzzer_tone': None,
            'behavior': 'neural_control',
            'intelligence_level': self.prediction_accuracy,
            'curiosity_level': curiosity_level,
            'total_experiences': self.total_experiences
        }
    
    async def _learn_from_experience(self, current_sensors: torch.Tensor):
        """Learn from the previous action's outcome"""
        
        # Predict what should have happened
        with torch.no_grad():
            predicted_outcome = self.prediction_net(self.previous_sensor_state, self.previous_action)
        
        # Calculate prediction error (learning signal)
        prediction_error = torch.nn.functional.mse_loss(predicted_outcome, current_sensors)
        
        # Calculate curiosity reward (higher error = more interesting)
        curiosity_reward = min(prediction_error.item(), 1.0)
        
        # Store experience
        experience = Experience(
            timestamp=time.time(),
            sensor_state=self.previous_sensor_state.cpu(),
            action_taken=self.previous_action.cpu(),
            outcome_sensors=current_sensors.cpu(),
            prediction_error=prediction_error.item(),
            curiosity_reward=curiosity_reward
        )
        
        self.experience_buffer.add_experience(experience)
        self.total_experiences += 1
        
        # Learning step if we have enough experiences
        if len(self.experience_buffer.experiences) >= 32:
            await self._neural_learning_step()
    
    async def _neural_learning_step(self):
        """Perform one learning step on recent experiences"""
        
        batch = self.experience_buffer.get_recent_batch(32)
        if batch is None:
            return
        
        # Train prediction network
        self.optimizer.zero_grad()
        
        # Forward pass
        predictions = self.prediction_net(batch['sensor_states'], batch['actions'])
        
        # Loss calculation
        prediction_loss = nn.functional.mse_loss(predictions, batch['outcomes'])
        
        # Curiosity-weighted loss (learn more from surprising experiences)
        curiosity_weights = batch['curiosity_rewards'] / (batch['curiosity_rewards'].sum() + 1e-8)
        weighted_loss = prediction_loss * curiosity_weights.mean()
        
        # Backpropagation
        weighted_loss.backward()
        self.optimizer.step()
        
        # Update metrics
        self.prediction_accuracy = 1.0 - min(prediction_loss.item(), 1.0)
        self.learning_rate = weighted_loss.item()
        
        # Log progress
        if self.total_experiences % 100 == 0:
            print(f"🧠 Learning update: {self.total_experiences} experiences, "
                  f"accuracy: {self.prediction_accuracy:.3f}, "
                  f"learning: {self.learning_rate:.4f}")
    
    def _calculate_curiosity(self, sensor_data: Dict[str, Any]) -> float:
        """Calculate curiosity level for current situation"""
        
        # Simple curiosity based on prediction confidence
        base_curiosity = 1.0 - self.prediction_accuracy
        
        # Boost curiosity for novel situations
        distance = sensor_data.get('ultrasonic_distance', 100)
        if distance < 30 or distance > 80:  # Unusual distances
            base_curiosity *= 1.5
        
        battery = sensor_data.get('battery_voltage', 12.0)
        if battery < 11.5:  # Low battery situations
            base_curiosity *= 0.5  # Less curious when survival threatened
        
        return min(base_curiosity, 1.0)
    
    def get_status(self) -> Dict[str, Any]:
        """Get current neural interface status"""
        return {
            'initialized': True,
            'device': self.device,
            'total_experiences': self.total_experiences,
            'prediction_accuracy': self.prediction_accuracy,
            'learning_rate': self.learning_rate,
            'buffer_size': len(self.experience_buffer.experiences),
            'intelligence_level': self.prediction_accuracy
        }