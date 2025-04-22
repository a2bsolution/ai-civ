#!/bin/bash

# activate the virtual environment
source env/bin/activate

# set the environment variables
export FLASK_APP=app.py
export FLASK_DEBUG=1

# start the Flask application
gunicorn --bind=0.0.0.0:8003 --timeout 600 app:app
