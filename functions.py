import numpy as np
import pandas as pd 
import cv2
import copy
import tf_keras as keras
import os
import re

from datetime import datetime
from PyPDF2 import PdfWriter, PdfReader
from pdf2image import convert_from_path
from PIL import Image
from tensorflow.keras.layers import BatchNormalization


carrier_model = keras.models.load_model('models/civ_page_8.h5')
ALLOWED_EXTENSIONS = set(['pdf', 'png', 'jpg', 'jpeg', 'bmp', 'docx', 'xlsx', 'xls','tiff'])


def special_char_filter(filename):
    """
    This function receives a file name string, usually an invoice number, and removes special characters.
    """
    return re.sub('[^A-Za-z0-9]+', '', filename)

def table_remove_null(table):
    indexes = sorted([i for i,x in enumerate(table['goods_description']) if x is None], reverse=True)
    for col in table:
        for index in indexes:
            del table[col][index]
    return table

def table_filter(table):
    table = table_remove_null(table)
    table['price'] = list(map(clean_amount, table['price']))
    if 'unit_price' in table:
        table['unit_price'] = list(map(clean_amount, table['unit_price']))
    return table

def table_kobe(table):
    result = []
    for item in table['product_code']:
        # Use regex to find text inside parentheses
        matches = re.findall(r'\((.*?)\)', item)
        if matches:
            result.append(matches[0])
    table['product_code'] = result
    return table

def table_zhong(table):
    first_value = next((item for item in table['goods_description'] if item is not None), "")
    length = max(len(v) for v in table.values())
    table['goods_description'] = [first_value] * length
    return table

def words_in_string(word_list, a_string):
    return set(word_list).intersection(a_string.split())

def temp_currency_filter(currency):
    currency = re.sub('[^A-Za-z ]+', '', currency).upper()

    if currency == "US":
        return "USD"
    curr_list = ['AUD', 'USD', 'EUR', 'JPY']
    if currency:
        for word in words_in_string(curr_list, currency.upper()):
            return word
            break


def clean_currency(value):
    """Standardizes currency values to ISO 4217 codes."""
    currency_map = {
        "$": "USD", "A$": "AUD", "¥": "JPY", "NZ$": "NZD", "€": "EUR",
    }
    
    if pd.isna(value) or value == "":
        return ""
    
    # If currency is already in ISO format, return as-is
    if value in currency_map.values():
        return value
    
    # Match known symbols
    for symbol, iso in currency_map.items():
        if symbol in value:
            return iso
    
    return value  # Default return

def clean_amount(value):
    """Standardizes the amount column to a float with 2 decimal places."""
    if value is None or str(value).strip() == "":
        return None
    
    # Remove currency symbols and non-numeric characters
    value = re.sub(r'[^\d.]', '', str(value))
        
    try:
        return float(value)
    except ValueError:
        return None  # Return None for corrupt entries

def container_separate(containers, pattern="[a-zA-Z]{4}[0-9]{7}"):
    """
    This function receives a string containing all containers and outputs them into a list
    of containers following standardized formatting.
    """
    formatted_container = re.findall(pattern, re.sub('[^A-Za-z0-9]+','', containers))
    return formatted_container

def file_name(filename):
    return filename.rsplit('.', 1)[0]

def file_ext(filename):
    return filename.rsplit('.', 1)[1].lower()

def allowed_file(filename):
    return '.' in filename and file_ext(filename) in ALLOWED_EXTENSIONS

def convert_date(timestamp):
    d = datetime.utcfromtimestamp(timestamp)
    formated_date = d.strftime('%d %b %Y %H:%M')
    return formated_date

def img_preprocess(image, image_size):
    im = image.resize((image_size,image_size))
    im = np.array(im)/255
    im = np.expand_dims(im, axis=0)
    return im

def invoice_page(image):
    image = img_preprocess(image, 224)
    pred=page_model.predict(image)
    return round(pred[0][0])

def classify_page(image):
    image = img_preprocess(image, 256)
    return np.argmax(carrier_model.predict(image))