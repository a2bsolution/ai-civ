import numpy as np
import pandas as pd 
import cv2
import copy
import os
import re

from datetime import datetime
from PyPDF2 import PdfWriter, PdfReader
from pdf2image import convert_from_path
from PIL import Image

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
    table['goods_description'] = [val.replace("\n", " ") if isinstance(val, str) else val for val in table['goods_description']]
    if 'unit_price' in table:
        table['unit_price'] = list(map(clean_amount, table['unit_price']))
    if 'unit_quantity' in table:
        table['unit_quantity'] = list(map(unit_filter, table['unit_quantity']))
    return table

def table_kobe(table):
    result = []
    for item in table['product_code']:
        # Use regex to find text inside parentheses
        matches = re.findall(r'\((.*?)\)', item)
        if matches:
            result.append(matches[0])
    table['product_code'] = result
    table['goods_description'] = remove_jp_characters(table['goods_description'])
    
    # Prepend product_code to goods_description
    for i in range(len(table['goods_description'])):
        code = table['product_code'][i]
        desc = table['goods_description'][i]
        if code is not None:
            table['goods_description'][i] = f"{code} - {desc}"

    return table

def remove_jp_characters(description):
    # Assuming `table` is your defaultdict
    cleaned_goods_description = []

    for desc in description:
        # Remove Japanese characters: Hiragana (\u3040–\u309F), Katakana (\u30A0–\u30FF), Kanji (\u4E00–\u9FBF)
        desc = re.sub(r'[\u3040-\u30FF\u4E00-\u9FFF]', '', desc)
        # Remove corrupted OCR-like artifacts (e.g. [ f& #x ], [ fe#x], etc.)
        desc = re.sub(r'\[\s*[^]]+\]', '', desc)
        # remove stray square brackets left behind
        desc =  re.sub(r'[\[\]【】]', '', desc)
        # Remove empty or whitespace-only parentheses: (), (   )
        desc = re.sub(r'\(\s*\)', '', desc)
        cleaned_goods_description.append(desc.strip())

    # Replace the old list
    return cleaned_goods_description

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

        # Remove rows where both keys are missing or null
        pkl_subset = pkl_subset.dropna(subset=['product_code', 'goods_description'])

        # Group by the merge keys and take the first wp value (or use other aggregation logic)
        pkl_subset = pkl_subset.groupby(['product_code', 'goods_description'], as_index=False).agg({'wp': 'first'})

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

def unit_filter(unit):
    try:
        unit = re.sub('[^A-Za-z ]+', '', unit).upper()
        unit_list = ['PCS', 'PC', 'SETS', 'SET']
        if unit:
            for word in words_in_string(unit_list, unit.upper()):
                return word
                break
    except Exception as e:
        print(e)
        return unit


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

def clean_amount(value, strict=True):
    """
    Converts a value to float.
    
    - If strict=True: removes all non-digit and non-dot characters (negatives not allowed).
    - If strict=False: keeps minus sign and handles formats like '-$1,200.50'.
    """
    if value is None or str(value).strip() == "":
        return None

    try:
        val_str = str(value).strip()
        if strict:
            # Strict: only digits and period
            val_str = re.sub(r"[^\d.]", "", val_str)
        else:
            # Loose: keep digits, dot, and minus sign
            val_str = re.sub(r"[^\d\.-]", "", val_str)
        return float(val_str)
    except ValueError:
        return value

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