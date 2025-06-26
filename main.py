"""
Main entry point for the brain server
"""
import asyncio
import signal
import sys
from brain_server import BrainServer

def signal_handler(signum, frame):
    """Handle shutdown gracefully"""
    print("\n🧠 Brain server shutting down...")
    sys.exit(0)

async def main():
    """
    Main function
    """
    # Handle Ctrl+C gracefully
    signal.signal(signal.SIGINT, signal_handler)
    
    print("🧠 Initializing brain server...")
    
    try:
        # Create and start brain server
        brain_server = BrainServer()
        await brain_server.start_server()
        
    except Exception as e:
        print(f"🧠 Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    print("🧠 Starting brain server...")
    asyncio.run(main())