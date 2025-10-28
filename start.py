#!/usr/bin/env python3
import os
import subprocess
import sys

# Get PORT from environment, default to 8000 if not set
port = os.environ.get('PORT', '8000')

# Ensure port is a valid integer
try:
    port_int = int(port)
except (ValueError, TypeError):
    port = '8000'
    print(f"Invalid PORT environment variable, defaulting to {port}")

# Start uvicorn with the correct port
cmd = [
    'uvicorn', 
    'server:app', 
    '--host', '0.0.0.0', 
    '--port', str(port)
]

print(f"Starting server on port {port}")
subprocess.run(cmd)