from flask import Flask, jsonify, request
import shutil, json, requests

from extract import predict
import os
from datetime import datetime
from ast import literal_eval

"""import logging
from logging.handlers import RotatingFileHandler"""

# Allowed extension you can set your own

app = Flask(__name__)

#app routes will have to be for user interface of table extraction and learning
@app.route("/")
def home():
    return "Server Live"

@app.route("/compare", methods = ['POST'])
def compare():
    """
    Receives files, extracts data and pushes it to the API endpoint for comparison.
    """
    predictions = {}
    params = request.args
    if (params == None):
        params = request.args

    # if parameters are found, return a prediction
    if (params != None):
        #try:
        user_id = request.form['user_id']
        process_id = literal_eval(request.form['process_id'])
        uploaded_by = request.form['uploadedby']
        date_uploaded = request.form['dateuploaded']
        shipment_num = request.form['shipment_num']

        if "file" in request.form:
            file_url = request.form['file']
            response = requests.get(file_url)
            filename = file_url.rsplit('/', 1)[1]
            if response.status_code != 200:
                raise ValueError(f"Failed to fetch file from URL: {file_url} (Status: {response.status_code})")
            if not response.content:
                raise ValueError(f"Downloaded file is empty: {file_url}")
            else:
                return predict(response.content, filename, process_id, user_id , uploaded_by, date_uploaded, shipment_num, file_url)
        else:
            file_obj = request.files.getlist('file')[0]
            if file_obj.filename != '':
                return predict(file_obj.read(), file_obj.filename, process_id, user_id, uploaded_by, date_uploaded, shipment_num)
            else:
                return "No file submitted."

        """except Exception as ex:
                                    return str(ex)"""

if __name__ == '__main__':
    #alogging.basicConfig(handlers=[RotatingFileHandler('logs/error.log', maxBytes=100000, backupCount=10)], level=logging.WARNING, format='%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    app.run(debug=True)