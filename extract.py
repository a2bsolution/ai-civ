import os, shutil, json, glob, re, io, cv2, collections #keras
import pandas as pd
import numpy as np
import pymssql
from pymssql import _mssql
from pdf2image import convert_from_bytes
from PyPDF2 import PdfMerger
from PIL import Image
from collections import defaultdict

from functions import * 

from azure.core.exceptions import ResourceNotFoundError
from azure.ai.formrecognizer import DocumentAnalysisClient, AnalyzeResult
from azure.core.credentials import AzureKeyCredential

endpoint = "https://ai-cargomation.cognitiveservices.azure.com/"
credential = AzureKeyCredential("a6a3fb5f929541648c788d45e6566603")
document_analysis_client = DocumentAnalysisClient(endpoint, credential)
default_model_id = "civ2_2"
data_folder = "../ai-data/test-ftp-folder/"
#data_folder = "E:/A2BFREIGHT_MANAGER/"
#poppler_path = r"C:\Program Files\poppler-21.03.0\Library\bin"

civ_indices = {'DEANS': 0, 'EMEDCO': 1, 'FALSE': 2, 'NB1': 3, 'NB2': 4, 'NINGBO': 5}
model_ids = {0: "civ_deans", 1: "civ2_2" }
carrier_data = {"NB": {"model_1": "civ_nb1", "model_2": "civ_nb2"}}

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

def form_recognizer_filter(result):
    prediction={}
    table = defaultdict(list)

    for analyzed_document in result.documents:
        #print("Document was analyzed by model with ID {}".format(result.model_id))
        #print("Document has confidence {}".format(analyzed_document.confidence))
        for name, field in analyzed_document.fields.items():
            if name=='table':
                #print("Field '{}' ".format(name))
                for row in field.value:
                    row_content = row.value
                    for key, item in row_content.items():
                        #print('Field {} has value {}'.format(key, item.value))
                        if key == "origin" and item.value:
                            table[key].append(special_char_filter(item.value))
                        else:
                            table[key].append(item.value)
            else:
                prediction[name]=field.value
                #print("Field '{}' has value '{}' with confidence of {}".format(name, field.value, field.confidence))

        prediction['table'] = table

    return prediction

def form_recognizer_one(file_name, page_num, model_id=default_model_id, document="", url=""):
    if document:
        poller = document_analysis_client.begin_analyze_document(model_id=model_id, document=document, pages=page_num)
    else:
        poller = document_analysis_client.begin_analyze_document_from_url(model_id=model_id, document_url=url, pages=page_num)

    prediction = form_recognizer_filter(poller.result())

    prediction['filename'] = file_name
    prediction['page'] = page_num

    return prediction

def multipage_combine(prediction_mult, shared_invoice, pdf_merge = False):
    """
    Receives json file of predictions and dict of shared invoices and 
    combines all pages with the same invoice number.
    """
    try:
        res = {}
        for i, v in shared_invoice.items():
            res[v] = [i] if v not in res.keys() else res[v] + [i]
        merged_predictions = {}
        for invoice_num, pages in res.items():
            page_nums = []
            new_file_name = special_char_filter(invoice_num) + "." +file_ext(pages[0])
            for idx, page in enumerate(pages):
                if idx==0:
                    merged_predictions[new_file_name] = prediction_mult[page].copy()
                    page_nums.append(prediction_mult[page]['page'])
                else:
                    for field, item_value in prediction_mult[page].items():
                        if field == 'table':
                            for table_key, table_value in item_value.items():
                                merged_predictions[new_file_name]['table'][table_key].extend(table_value)
                        elif field == 'page':
                            page_nums.append(item_value)
                        elif merged_predictions[new_file_name][field] is None and item_value is not None:
                            merged_predictions[new_file_name][field] = item_value
            merged_predictions[new_file_name]['page'] = page_nums
        return merged_predictions
    except Exception as ex:
        return str(ex)

def multipage_extraction(classifier_count, classification_page_2, classification_page_1, carrier_data, filename, predictions, shared_invoice, file_url="", file_bytes=""):
    last_page = classifier_count[classification_page_2][-1]
    for idx, page in enumerate(sorted(classifier_count[classification_page_1],reverse=True)):
        split_file_name = file_name(filename) + "_" + str(idx)
        split_file_name_ext = split_file_name + "." + file_ext(filename)

        if file_url:
            predictions[split_file_name_ext] = form_recognizer_one(filename, page, carrier_data["model_1"], url=file_url)
        else:
            predictions[split_file_name_ext] = form_recognizer_one(document=file_bytes, file_name=filename, page_num=page, model_id=carrier_data["model_1"])

        invoice_num = predictions[split_file_name_ext]['invoice_number']
        shared_invoice[split_file_name_ext] = invoice_num
        #predictions[split_file_name_ext]['table']['charge_description'] = charge_description_filter(predictions[split_file_name_ext]['table']['charge_description'], carrier_data["charge_description_filter"])
        for x in classifier_count[classification_page_2]:
            if page < x <= last_page:
                page_two_name = split_file_name+"_"+str(x) +"."+file_ext(filename)
                predictions[page_two_name] = form_recognizer_one(filename, x, carrier_data["model_2"], url=file_url, document=file_bytes)
                shared_invoice[page_two_name] = invoice_num
                #predictions[page_two_name]['table']['charge_description'] = charge_description_filter(predictions[page_two_name]['table']['charge_description'], carrier_data["charge_description_filter"])
        last_page = page
    return predictions, shared_invoice

def query_webservice_user(webservice_user):
    #query the server
    conn = pymssql.connect('a2bserver.database.windows.net', 'A2B_Admin', 'v9jn9cQ9dF7W', 'a2bcargomation_db')
    cursor = conn.cursor(as_dict=True)
    cursor.execute("SELECT TOP (1) * FROM [dbo].[user_webservice] WHERE [user_id] = %s", webservice_user)
    user_query=cursor.fetchone()
    cursor.close()
    return user_query

def add_webservice_user(predictions, file, user_query):
    predictions[file]["webservice_link"] = user_query['webservice_link'] #"https://a2btrnservices.wisegrid.net/eAdaptor"  
    predictions[file]["webservice_username"] =  user_query['webservice_username']#"A2B"
    predictions[file]["webservice_password"] = user_query['webservice_password']#"Hw7m3XhS"
    predictions[file]["server_id"] = user_query['server_id']# "TRN"
    predictions[file]["enterprise_id"] =  user_query['enterprise_id'] #"A2B"
    predictions[file]["company_code"] = user_query['company_code']#"SYD"
    return predictions

def push_parsed_inv(predictions, process_id, user_id, uploaded_by, date_uploaded, shipment_num, filename="", file_url="", num_pages=None):
    conn = pymssql.connect('a2bserver.database.windows.net', 'A2B_Admin', 'v9jn9cQ9dF7W', 'a2bcargomation_db')
    #conn = pymssql.connect('a2bserver.database.windows.net', 'A2B_Admin', 'v9jn9cQ9dF7W', 'dev.a2bcargomation_db')
    cursor = conn.cursor(as_dict=True)
    cursor.execute("""
        IF EXISTS (SELECT TOP 1 1 FROM [dbo].[document_upload_compile] WHERE [process_id]=%s)
            BEGIN
            UPDATE [dbo].[document_upload_compile] SET [parsed_inv]=%s WHERE [process_id]=%s
            END
        ELSE
            BEGIN
            INSERT INTO [dbo].[document_upload_compile] (user_id, process_id, filename, filepath, dateuploaded, uploadedby, status, num_pages, parsed_inv, shipment_num) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            END
        """,
        (process_id, predictions, process_id, user_id, process_id, filename, file_url, date_uploaded, uploaded_by, "processing", num_pages, predictions, shipment_num)) 

    conn.commit()
    cursor.close()


def predict(file_bytes, filename, process_id, user_id, uploaded_by, date_uploaded, shipment_num, file_url=""):
    predictions = {}
    shared_invoice = {}
    classifier_count = collections.defaultdict(list)

    user_query = query_webservice_user(user_id)
    ext = file_ext(filename)

    if ext == "pdf":
        images = convert_from_bytes(file_bytes, grayscale=True, fmt="jpeg") #, poppler_path=poppler_path
        inputpdf = PdfReader(io.BytesIO(file_bytes), strict=False)
        if inputpdf.is_encrypted:
            try:
                inputpdf.decrypt('')
                #print('File Decrypted (PyPDF2)')
            except:
                print("Decryption error")
        
        #classify and split
        for page, image in enumerate(images):
            pred = classify_page(image)
            if pred == civ_indices['FALSE']:
                pass
            else:
                page_num = page+1 #for counters to begin at 1
                split_file_name = file_name(filename) +"_pg"+str(page_num)+".pdf"
                split_file_path = data_folder+"SPLIT/"+split_file_name
                if pred == civ_indices['NB1']:
                    classifier_count['NB1'].append(page_num)
                elif pred == civ_indices['NB2']:
                    classifier_count['NB2'].append(page_num)
                else:
                    if file_url:
                        predictions[split_file_name] = form_recognizer_one(url=file_url, file_name=filename, page_num=page_num, model_id=model_ids[pred])
                    else:
                        predictions[split_file_name] = form_recognizer_one(document=file_bytes, file_name=filename, page_num=page_num, model_id=model_ids[pred])
                    shared_invoice[split_file_name] = predictions[split_file_name]['invoice_number']
                    predictions[split_file_name]['table'] = table_remove_null(predictions[split_file_name]['table'])

        if classifier_count.get("NB2") and classifier_count.get("NB1"):
            predictions, shared_invoice = multipage_extraction(classifier_count, "NB2", "NB1", carrier_data["NB"], filename, predictions, shared_invoice, file_bytes=file_bytes)

    elif ext in ["jpg", "jpeg", "png",'.bmp','.tiff']:
        pil_image = Image.open(io.BytesIO(file_bytes)).convert('L').convert('RGB') 
        if invoice_page(pil_image) == 1:
            predictions[filename] = form_recognizer_one(document=file_bytes, file_name=filename, page_num=1, model_id=default_model_id)
            predictions[filename]['table'] = table_remove_null(predictions[filename]['table'])
    else:
        return "File type not allowed."


    if len(predictions) > 1:
            predictions = multipage_combine(predictions, shared_invoice)
    else:
        for file in predictions:
            predictions[file]['page'] = [predictions[file]['page']]

    for file in predictions:
        predictions = add_webservice_user(predictions, file, user_query)
        predictions[file]['process_id'] = process_id
        predictions[file]['user_id'] = user_id
        payload = {
                "user_id": user_id, 
                "jsonstring": json.dumps(predictions[file])
                }

        push_parsed_inv(json.dumps(predictions[file]), process_id, user_id, uploaded_by, date_uploaded, shipment_num, filename, file_url, len(predictions[file]['page']))

    return payload