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

        if "file" in request.form:
            file_urls = literal_eval(request.form['file'])
            #return extract_compare((file_urls), user_id, process_id)
            for idx, url in enumerate(file_urls):
                response = requests.get(url)
                filename = url.rsplit('/', 1)[1]
                predictions = predict(response.content, filename, process_id[idx], user_id)
            return predictions
        else:
            file_obj = request.files.getlist('file')
            if file_obj[0].filename != '':
                for idx, obj in enumerate(file_obj):
                    predictions = predict(obj.read(), obj.filename, process_id[idx], user_id)
                return predictions
            else:
                return "No file submitted."

        """except Exception as ex:
                                    return str(ex)"""

if __name__ == '__main__':
    #alogging.basicConfig(handlers=[RotatingFileHandler('logs/error.log', maxBytes=100000, backupCount=10)], level=logging.WARNING, format='%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    app.run(debug=True)