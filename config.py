"""
Configuration for the brain server
"""
import os

# Server configuration
SERVER_HOST = os.getenv('BRAIN_HOST', 'localhost')
SERVER_PORT = int(os.getenv('BRAIN_PORT', 9977))

# Neural network configuration (placeholders for now)
NEURAL_DEVICE = 'cuda' if os.getenv('USE_GPU', 'true').lower() == 'true' else 'cpu'
NEURAL_UPDATE_RATE = 10  # Hz

# Robot connection settings
MAX_ROBOTS = int(os.getenv('MAX_ROBOTS', 10))
CONNECTION_TIMEOUT = 30  # seconds

# Logging
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')

print(f"🧠 Brain server config loaded:")
print(f"   Server: {SERVER_HOST}:{SERVER_PORT}")
print(f"   Neural device: {NEURAL_DEVICE}")
print(f"   Max robots: {MAX_ROBOTS}")