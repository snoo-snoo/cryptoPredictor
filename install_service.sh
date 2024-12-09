#!/bin/bash

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "Created virtual environment"
fi

# Activate virtual environment
source venv/bin/activate

# Install requirements
pip install -r requirements.txt
echo "Installed requirements"

# Create logs directory if it doesn't exist
mkdir -p logs
echo "Created logs directory"

# Create opportunities directory if it doesn't exist
mkdir -p opportunities
echo "Created opportunities directory"

# Copy service file to systemd directory
sudo cp trading_bot.service /etc/systemd/system/
echo "Copied service file"

# Reload systemd daemon
sudo systemctl daemon-reload

# Enable the service to start on boot
sudo systemctl enable trading_bot

# Start the service
sudo systemctl start trading_bot

# Check status
sudo systemctl status trading_bot 