#!/bin/bash

echo "-------Start the memcached service"
nohup /usr/bin/supervisord -c /etc/supervisor/supervisord.conf &

echo "-------Start the fastAPI server"
fastapi run main.py --port 8000