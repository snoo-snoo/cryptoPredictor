#!/bin/bash

# Change to the bot directory
cd /home/pi/trading_bot

# Activate virtual environment
source venv/bin/activate

# Run the bot with specified trading pairs
python3 trading_bot.py >> logs/cron.log 2>&1