from azure.ai.documentintelligence.models import AnalyzeResult
from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient
from flask import jsonify

def classify_document(file_bytes):
    classifications = {}
    """Classify a document using the Azure Document Intelligence client."""
    endpoint = "https://ai-cargomation.cognitiveservices.azure.com/"
    credential = AzureKeyCredential("a6a3fb5f929541648c788d45e6566603")
    client = DocumentIntelligenceClient(endpoint, credential)
    classifier_id = "classify_compile_3"#current_app.config["CLASSIFIER_ID"]

    try:

        poller = client.begin_classify_document(classifier_id, body=file_bytes, split="auto")
        result: AnalyzeResult = poller.result()

        if result.documents:
            for doc in result.documents:
                if doc.bounding_regions:
                    page_numbers = [region.page_number for region in doc.bounding_regions]
                    
                    # Check if the doc_type already exists in the classifications dictionary
                    if doc.doc_type in classifications:
                        classifications[doc.doc_type].extend(page_numbers)
                    else:
                        classifications[doc.doc_type] = page_numbers

        return classifications

    except Exception as e:
        # Handle errors and return a JSON response
        error_message = {"error": str(e)}
        print(f"Error during document classification: {error_message}")
        return jsonify(error_message), 500
