import numpy as np
import pandas as pd 
import cv2
import copy
import keras
import os
import re

from datetime import datetime
from PyPDF2 import PdfWriter, PdfReader
from pdf2image import convert_from_path
from PIL import Image
from tensorflow.keras.layers import BatchNormalization

ALLOWED_EXTENSIONS = set(['pdf', 'png', 'jpg', 'jpeg', 'bmp', 'docx', 'xlsx', 'xls','tiff'])


def special_char_filter(filename):
    """
    This function receives a file name string, usually an invoice number, and removes special characters.
    """
    return re.sub('[^A-Za-z0-9]+', '', filename)

def table_remove_null(table, column="goods_description"):
    indexes = sorted([i for i,x in enumerate(table[column]) if x is None], reverse=True)
    for col in table:
        for index in indexes:
            del table[col][index]
    return table

def table_filter(table, column="goods_description"):
    table = table_remove_null(table, column)
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

def table_htl(table, column="goods_description"):
    table = table_filter(table, column="product_code")
    indexes = sorted([i for i,x in enumerate(table["goods_description"]) if x is None], reverse=True)
    for index in indexes:
        table["goods_description"][index] = "SOFA"
    return table

def merge_by_invoice(shared_invoice):
    res = {}
    for i, v in shared_invoice.items():
        res[v] = [i] if v not in res.keys() else res[v] + [i]
    return res

def merge_wp_column_with_dataframe(data_dict, shared_invoice):
    res_map = merge_by_invoice(shared_invoice)
    for invoice_number, file_list in res_map.items():
        if len(file_list) < 2:
            continue  # No merging needed
        
        main_file, pkl_file = file_list
        main_table = pd.DataFrame(data_dict[main_file]['table'])

        pkl_subset = pd.DataFrame({
            'product_code': data_dict[pkl_file]['table']['product_code'],
            'goods_description': data_dict[pkl_file]['table']['goods_description'],
            'wp': data_dict[pkl_file]['table']['wp']
        })

        # Merge wp into main_table using product_code and goods_description
        merged_table = pd.merge(
            main_table,
            pkl_subset,
            on=['product_code', 'goods_description'],
            how='left'
        )

        # Update the original data dict with the merged table
        data_dict[main_file]['table'] = merged_table.to_dict(orient='list')
        data_dict[main_file]['classification'] = "kuka"

        # Drop the secondary file
        del data_dict[pkl_file]

    return data_dict

def table_kuka(table):
    #adds feather cushion at the beginning of the description
    indexes = sorted(
    [i for i, x in enumerate(table["extra_description"]) if x and x.strip().lower() == "feather cushion"],
    reverse=True)
    for index in indexes:
        table["goods_description"][index] = "Feather cushion " + table["goods_description"][index]
    return table

def table_zhong(table):
    first_value = next((item for item in table['goods_description'] if item is not None), "")
    length = max(len(v) for v in table.values())
    table['goods_description'] = [first_value] * length
    return table

def words_in_string(word_list, a_string):
    return set(word_list).intersection(a_string.split())

def temp_currency_filter(currency):
    try:
        currency = re.sub('[^A-Za-z ]+', '', currency).upper()

        if currency == "US":
            return "USD"
        curr_list = ['AUD', 'USD', 'EUR', 'JPY']
        if currency:
            for word in words_in_string(curr_list, currency.upper()):
                return word
                break
    except Exception as e:
        print(e)
        return currency


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
    """Standardizes the value to a float with 2 decimal places."""
    if value is None or str(value).strip() == "":
        return None
    
    try:
        return float(re.sub(r'[^\d.]', '', str(value)))
    except ValueError:
        return value  # Return None for corrupt entries

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
