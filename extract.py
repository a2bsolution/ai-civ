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
from excel import *
from classify import classify_document

from azure.core.exceptions import ResourceNotFoundError
from azure.core.credentials import AzureKeyCredential

from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import AnalyzeResult, AnalyzeDocumentRequest

endpoint = "https://ai-cargomation.cognitiveservices.azure.com/"
credential = AzureKeyCredential("a6a3fb5f929541648c788d45e6566603")
document_analysis_client = DocumentIntelligenceClient(endpoint, credential)
default_model_id = "civ2_2"
data_folder = "../ai-data/test-ftp-folder/"

model_ids = {'deans': "civ_deans", "dongguan_sunup":"civ_dongguan_sunup", 'emedco': "civ2_2", "htl":"civ_htl_1",'kobe': "civ_kobe_neural_5",'kuka': "civ_kuka", 'kuka_pkl': "civ_kuka_pkl", "nb1": "civ_nb1", "nb2": "civ_nb2", "premier_intl":"civ_premiere_1", "premier_intl_pkl":"civ_premier_pkl",'starway': "civ_starway_1", 'white_feathers': "wf_neural_2", 'zhong_shen': "civ_zhong_shen", "shanewsha":"civ_shanewsha_1", "shanewsha_pkl":"pkl_shanewsha"}
carrier_data = {"NB": {"nb1": "civ_nb1", "nb2": "civ_nb2"}}


def form_recognizer_filter(result):
    prediction={}
    table = defaultdict(list)

    if result.documents:
        for analyzed_document in result.documents:
            #print("Document was analyzed by model with ID {}".format(result.model_id))
            #print("Document has confidence {}".format(analyzed_document.confidence))
            for name, field in analyzed_document.fields.items():
                if name=='table' and field.value_array:
                    for row in field.value_array:
                        row_value = row['valueObject']
                        for key, item in row_value.items():
                            field_value = item.get("valueString") if item.get("valueString") else item.content
                            #print('Field {} has value {}'.format(key, field_value))
                            if key == "origin" and field_value:
                                table[key].append(special_char_filter(field_value))
                            else:
                                table[key].append(field_value)
                elif name == "incoterm":
                    field_value = field.get("valueString") if field.get("valueString") else field.content
                    if field_value:
                        prediction[name] = special_char_filter(field_value)
                elif name == "container":
                    field_value = field.get("valueString") if field.get("valueString") else field.content
                    if field_value:
                        prediction[name] = container_separate(field_value)
                elif name == "total_weight":
                    field_value = field.get("valueString") if field.get("valueString") else field.content
                    if field_value:
                        prediction[name] = clean_amount(field_value)
                elif name == "discount":
                    field_value = field.get("valueString") if field.get("valueString") else field.content
                    if field_value:
                        prediction[name] = clean_amount(field_value)
                else:
                    field_value = field.get("valueString") if field.get("valueString") else field.content
                    prediction[name] = field_value
                    #print("Field '{}' has value '{}' with confidence of {}".format(name, field_value, field.confidence))

            prediction['table'] = table

    return prediction

def form_recognizer_one(file_name, page_num, model_id=default_model_id, document="", url=""):
    page_num_form = ",".join(map(str, page_num)) #form recognizer format

    if document:
        poller = document_analysis_client.begin_analyze_document(model_id=model_id, body=document, pages=page_num_form)
    else:
        poller = document_analysis_client.begin_analyze_document(model_id=model_id, body={"urlSource": url}, pages=page_num_form)

    result: AnalyzeResult = poller.result()
    prediction = form_recognizer_filter(result)

    prediction['filename'] = file_name
    prediction['page'] = page_num
    return prediction

def multipage_combine(prediction_mult, shared_invoice, pdf_merge = False):
    """
    Receives json file of predictions and dict of shared invoices and 
    combines all pages with the same invoice number.
    """
    try:
        res = merge_by_invoice(shared_invoice)
        merged_predictions = {}
        for invoice_num, pages in res.items():
            page_nums = []
            new_file_name = special_char_filter(invoice_num) + "." +file_ext(pages[0])
            for idx, page in enumerate(pages):
                if idx==0:
                    merged_predictions[new_file_name] = prediction_mult[page].copy()
                    page_nums.extend(prediction_mult[page]['page'])
                else:
                    for field, item_value in prediction_mult[page].items():
                        if field == 'table':
                            for table_key, table_value in item_value.items():
                                merged_predictions[new_file_name]['table'][table_key].extend(table_value)
                        elif field == 'page':
                            page_nums.extend(item_value)
                        if field not in merged_predictions[new_file_name] or merged_predictions[new_file_name][field] is None:
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
    conn = pymssql.connect('a2bserver.database.windows.net', 'A2B_Admin', 'v9jn9cQ9dF7W', 'a2bcargomation_db')
    cursor = conn.cursor(as_dict=True)
    cursor.execute("SELECT TOP (1) * FROM [dbo].[user_webservice] WHERE [user_id] = %s", webservice_user)
    query = cursor.fetchone()
    if query is None:
        cursor.execute("SELECT TOP (1) * FROM [dbo].[vrpt_subaccount] WHERE [user_id] = %s", webservice_user)
        query = cursor.fetchone()
        cursor.execute("SELECT TOP (1) * FROM [dbo].[user_webservice] WHERE [user_id] = %s", query["account_id"])
        query=cursor.fetchone()
    cursor.close()
    return query

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
            UPDATE [dbo].[document_upload_compile] SET [parsed_inv]=%s, [num_pages]=%s  WHERE [process_id]=%s
            END
        ELSE
            BEGIN
            INSERT INTO [dbo].[document_upload_compile] (user_id, process_id, filename, filepath, dateuploaded, uploadedby, status, num_pages, parsed_inv, shipment_num) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            END
        """,
        (process_id, predictions, num_pages, process_id, user_id, process_id, filename, file_url, date_uploaded, uploaded_by, "processing", num_pages, predictions, shipment_num)) 

    conn.commit()
    cursor.close()


def predict(file_bytes, filename, process_id, user_id, uploaded_by, date_uploaded, shipment_num, file_url=""):
    predictions = {}
    shared_invoice = {}
    classifier_count = collections.defaultdict(list)

    user_query = query_webservice_user(user_id)
    ext = file_ext(filename)

    if ext in ["pdf", "jpg", "jpeg", "png",".bmp",".tiff", ".heif"]:
        classification = classify_document(file_bytes)
        print(classification)
        for key, pages in classification.items():
            split_file_name = file_name(filename) +"_pg"+str(pages)+"."+ext

            if key in ["false", "htl_t&c"]:
                pass
            elif key == 'nb1':
                classifier_count['NB1'].append(pages)
            elif key == 'nb2':
                classifier_count['NB2'].append(pages)
            else:
                if file_url:
                    predictions[split_file_name] = form_recognizer_one(url=file_url, file_name=filename, page_num=pages, model_id=model_ids[key])
                else:
                    predictions[split_file_name] = form_recognizer_one(document=file_bytes, file_name=filename, page_num=pages, model_id=model_ids[key])

                if key == 'kobe':
                    predictions[split_file_name]['currency'] = clean_currency(predictions[split_file_name]['total'])
                    predictions[split_file_name]['table'] = table_kobe(predictions[split_file_name]['table'])
                elif key == 'zhong_shen':
                    predictions[split_file_name]['currency'] = clean_currency(predictions[split_file_name]['total'])
                    predictions[split_file_name]['table'] = table_zhong(predictions[split_file_name]['table'])
                
                shared_invoice[split_file_name] = predictions[split_file_name]['invoice_number']

                if predictions[split_file_name].get('total'):
                    predictions[split_file_name]['total'] = clean_amount(predictions[split_file_name]['total'])

                if predictions[split_file_name].get('table'):
                    if key == "htl":
                        predictions[split_file_name]['table'] = table_htl(predictions[split_file_name]['table'])
                    else:
                        predictions[split_file_name]['table'] = table_filter(predictions[split_file_name]['table'])

        if classifier_count.get("NB2") and classifier_count.get("NB1"):
            predictions, shared_invoice = multipage_extraction(classifier_count, "NB2", "NB1", carrier_data["NB"], filename, predictions, shared_invoice, file_bytes=file_bytes)

        if classification.get("kuka") and classification.get("kuka_pkl"):
            predictions = merge_wp_column_with_dataframe(predictions, shared_invoice) #works if you assume civ only has one of each and not mulitple docs
            for file in predictions:
                if predictions[file]['classification'] == "kuka":
                    predictions[file]['table'] = table_kuka(predictions[file]['table'])
        elif classification.get("kuka"):
            for file in predictions:
                predictions[file]['table'] = table_kuka(predictions[file]['table'])


    elif ext == 'xlsx':
        predictions[filename] = extract_xlsx(file_bytes)

    elif ext == 'xls':
        predictions[filename] = extract_xls(pd.read_excel(io.BytesIO(file_bytes), engine='xlrd', sheet_name=0, header=None))

    else:
        return "File type not allowed."


    if len(predictions) > 1:
        predictions = multipage_combine(predictions, shared_invoice) #skips kuka since it just becomes one prediction 
    #else:
        #for file in predictions:
            #predictions[file]['page'] = [predictions[file]['page']]

    for file in predictions:
        if predictions[file]['currency'] is not None:
            predictions[file]['currency'] = temp_currency_filter(predictions[file]['currency'])
        predictions = add_webservice_user(predictions, file, user_query)
        predictions[file]['process_id'] = process_id
        predictions[file]['user_id'] = user_id
        payload = {
                "user_id": user_id, 
                "jsonstring": json.dumps(predictions[file])
                }

        push_parsed_inv(json.dumps(payload), process_id, user_id, uploaded_by, date_uploaded, shipment_num, filename, file_url, len(predictions[file]['page']))

    return payload