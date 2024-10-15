import pandas as pd
import re


def words_in_string(word_list, a_string):
    return set(word_list).intersection(a_string.split())

def temp_currency_filter(currency):
    currency = re.sub('[^A-Za-z]+', ' ', currency)
    curr_list = ['AUD', 'USD', 'EUR']

    if currency:
        for word in words_in_string(curr_list, currency.upper()):
            return word
            break

def extract_xlsx(df):
	prediction = {}
	prediction['supplier'] = df.iloc[0,3]
	prediction['invoice_number'] = df.iloc[11, 2]
	prediction['incoterm'] = str(df.iloc[10, 8]).split()[0] if pd.notna(df.iloc[10, 8]) else None
	prediction['page'] = 1

	# The table headers are in row 20 (index 19)
	header_row = 19  # Row 20 in zero-based index
	table_headers = df.iloc[header_row]  # Grab the headers from row 20
	table_df = df.iloc[header_row + 1:]  # Extract the data from the rows below the headers
	prediction['currency'] = temp_currency_filter(df.iloc[header_row, 4])

	table_df = table_df.iloc[:, :5] 
	custom_headers = ['product_code', 'goods_description', 'invoice_quantity', 'unit_price', 'price']
	table_df.columns = custom_headers

	table_df = table_df.dropna(how='all')  # Drop rows that are entirely empty

	# Stop reading the table when "Inland Charge USD" is encountered in the first column
	for index, row in table_df.iterrows():
		if isinstance(row.iloc[0], str) and "Inland Charge" in row.iloc[0]:
			table_df = table_df.loc[:index - 1]  # Only keep rows before this point
			break

	last_row_value = None
	if not table_df.empty:
		last_row_value = table_df.iloc[-1, 4]  # Get the value in column E (index 4) of the last row
		table_df = table_df.iloc[:-1]  # Drop the last row

	# Set the headers as column names of the table DataFrame
	table_headers = ['product_code', 'goods_description', 'invoice_quantity', 'unit_price', 'price']
	table_df.columns = table_headers
	table_df['unit_quantity'] = ["Piece"] * len(table_df['invoice_quantity'])

	prediction['table'] = table_df.to_dict(orient='list')
	prediction['total'] = round(last_row_value, 2)

	return prediction

def extract_xls(df):
	prediction = {}
	prediction['page'] = 1
	prediction['supplier'] = df.iloc[0].dropna().iloc[0].strip() if not df.iloc[0].dropna().empty else None
	prediction['invoice_number'] = None
	prediction['incoterm'] = None

	# Iterate through the DataFrame to find the necessary values
	for row in df.iterrows():
		row_idx, row_data = row
		for col_idx, value in enumerate(row_data):
			if isinstance(value, str):
				value_clean = value.strip().lower()

				if "invoice no" in value_clean:
					prediction['invoice_number'] = df.iloc[row_idx, col_idx + 1]  # Get the value next to "Invoice No."

				elif "price term" in value_clean:
					prediction['incoterm'] = df.iloc[row_idx, col_idx + 1]
					if pd.isna(prediction['incoterm']):
						prediction['incoterm'] = str(df.iloc[row_idx, col_idx + 2]).split()[0]
					else:
						prediction['incoterm'] = str(prediction['incoterm']).split()[0]

			# If both values are found, exit the loop early
		if prediction['invoice_number'] is not None and prediction['incoterm'] is not None:
			break

	# The table headers are in row 20 (index 19)
	item_no_row_index = df[df.apply(lambda row: row.astype(str).str.contains("ITEM NO", case=False, na=False).any(), axis=1)].index
	if not item_no_row_index.empty:
		header_row = item_no_row_index[0]
		table_headers = df.iloc[header_row]  # Grab the headers from row 20
		table_df = df.iloc[header_row + 1:]  # Extract the data from the rows below the headers
		prediction['currency'] = temp_currency_filter(df.iloc[header_row, 4])

		table_df = table_df.iloc[:, :5] 
		custom_headers = ['product_code', 'goods_description', 'invoice_quantity', 'unit_price', 'price']
		table_df.columns = custom_headers

		table_df = table_df.dropna(how='all')  # Drop rows that are entirely empty

		# Stop reading the table when "Inland Charge" is encountered in the first column
		for index, row in table_df.iterrows():
			if isinstance(row.iloc[0], str) and "Inland Charge" in row.iloc[0]:
				table_df = table_df.loc[:index - 1]  # Only keep rows before this point
				break

		last_row_value = None
		if not table_df.empty:
			last_row_value = table_df.iloc[-1, 4]  # Get the value in column E (index 4) of the last row
			table_df = table_df.iloc[:-1]  # Drop the last row

		# Set the headers as column names of the table DataFrame
		table_headers = ['product_code', 'goods_description', 'invoice_quantity', 'unit_price', 'price']
		table_df.columns = table_headers
		table_df['unit_quantity'] = ["Piece"] * len(table_df['invoice_quantity'])
		prediction['table'] = table_df.to_dict(orient='list')
		prediction['total'] = round(last_row_value, 2)

	else:
		raise ValueError("The header 'ITEM NO.:' could not be found in the file.")

	return prediction