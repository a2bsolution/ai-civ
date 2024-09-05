#!/bin/bash

# activate the virtual environment
source env/bin/activate

# set the environment variables
export FLASK_APP=app.py
export FLASK_DEBUG=1

# start the Flask application
flask run
