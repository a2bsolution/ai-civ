import pandas as pd
import re


def words_in_string(word_list, a_string):
    return set(word_list).intersection(a_string.split())

def temp_currency_filter(currency):
	currency = re.sub('[^A-Za-z ]+', '', currency)
	curr_list = ['AUD', 'USD', 'EUR']
	if currency:
		for word in words_in_string(curr_list, currency.upper()):
			return word
			break

def incoterm_filter(incoterm):
	incoterm = re.sub('[^A-Za-z ]+', '', incoterm).upper()
	incoterm_list = ['FOB', 'CFR', 'CIF', 'CIP', 'CPT', 'DAP', 'DAT', 'DDP', 'DPU', 'EXW', 'FAS', 'FC1', 'FC2', 'FCA']

	for word in incoterm_list:
		if word in incoterm:
			return word
			break
	return None

def extract_xlsx(df):
	prediction = {}
	prediction['page'] = [1]
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
					prediction['invoice_number'] = df.iloc[row_idx, col_idx + 1]
					if pd.isna(prediction['invoice_number']):
						remaining = re.sub(r'invoice\s*no\.?:?', '', value, flags=re.IGNORECASE)
						if remaining:
							prediction['invoice_number'] = remaining

				elif "price term" in value_clean:
					col_add = 1
					while col_add < 5:
						value = df.iloc[row_idx, col_idx + col_add] 
						if pd.notna(value) and value != '':
							prediction['incoterm'] = value
							break
						col_add += 1

					if pd.isna(prediction['incoterm']):
						prediction['incoterm'] = incoterm_filter(str(df.iloc[row_idx, col_idx + 2]).split()[0])
					else:
						prediction['incoterm'] = incoterm_filter(str(prediction['incoterm']).split()[0])

			# If both values are found, exit the loop early
		if prediction['invoice_number'] is not None and prediction['incoterm'] is not None:
			break

	item_no_row_index = df[df.apply(lambda row: row.astype(str).str.contains("ITEM NO", case=False, na=False).any(), axis=1)].index
	if not item_no_row_index.empty:
		header_row = item_no_row_index[0]
		table_headers = df.iloc[header_row]  # Grab the headers from row 20
		table_df = df.iloc[header_row + 1:]  # Extract the data from the rows below the headers

		try:
			prediction['currency'] = temp_currency_filter(df.iloc[header_row, 4])
			table_df = table_df.iloc[:, :5] 
			custom_headers = ['product_code', 'goods_description', 'invoice_quantity', 'unit_price', 'price']
			table_df.columns = custom_headers

			table_df = table_df.dropna(how='all')  # Drop rows that are entirely empty

			# Stop reading the table when "Inland Charge USD" or "Total" is encountered in the first column
			for index, row in reversed(list(table_df.iterrows())):
				if isinstance(row.iloc[0], str) and "inland charge" in row.iloc[0].lower(): #for some reason row points to row after inland charge
					table_df = table_df.loc[:index - 2]  # Only keep rows before this point, and drop the last row
					last_row_value = re.sub('[^0-9.]+', '', str(table_df.iloc[-1, 4]))  # Get the value in column E (index 4) of the last row
					prediction['total'] = round(float(last_row_value), 2)
					break
				elif isinstance(row.iloc[0], str) and "total" in row.iloc[0].lower(): #row points to total row
					table_df = table_df.loc[:index - 1]
					prediction['total'] = round(float(row.iloc[4]), 2)

			if not table_df.empty:
				table_df = table_df.dropna(subset=[table_df.columns[0], table_df.columns[1]])  # Remove NaN values
				table_df = table_df[table_df.iloc[:, 0].str.strip().astype(bool) & table_df.iloc[:, 1].str.strip().astype(bool)] # Remove rows where first or second column is empty or only spaces

			# Set the headers as column names of the table DataFrame
			table_headers = ['product_code', 'goods_description', 'invoice_quantity', 'unit_price', 'price']
			table_df.columns = table_headers
			table_df['unit_quantity'] = ["Piece"] * len(table_df['invoice_quantity'])

			prediction['table'] = table_df.to_dict(orient='list')
		except:
			return prediction

	return prediction

def extract_xls(df):
	prediction = {}
	prediction['page'] = [1]
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
					col_add = 1
					while col_add < 5:
						value = df.iloc[row_idx, col_idx + col_add] 
						if pd.notna(value) and value != '':
							prediction['incoterm'] = value
							break
						col_add += 1

					if pd.isna(prediction['incoterm']):
						prediction['incoterm'] = incoterm_filter(str(df.iloc[row_idx, col_idx + 2]).split()[0])
					else:
						prediction['incoterm'] = incoterm_filter(str(prediction['incoterm']).split()[0])

			# If both values are found, exit the loop early
		if prediction['invoice_number'] is not None and prediction['incoterm'] is not None:
			break

	item_no_positions = df.apply(lambda row: row.astype(str).str.contains("ITEM NO", case=False, na=False), axis=1)
	first_item_no_position = item_no_positions.stack()
	header_row, header_col = first_item_no_position[first_item_no_position].index[0]
	
	if not first_item_no_position.empty:
		table_headers = df.iloc[header_row]
		table_df = df.iloc[header_row + 1:]  # Extract the data from the rows below the headers
		prediction['currency'] = temp_currency_filter(df.iloc[header_row, 4])

		table_df = table_df.iloc[:, header_col:header_col+5] 
		custom_headers = ['product_code', 'goods_description', 'invoice_quantity', 'unit_price', 'price']
		table_df.columns = custom_headers
		table_df = table_df.dropna(how='all')  # Drop rows that are entirely empty

		for index, row in table_df.iterrows():
			if isinstance(row.iloc[0], str) and "inland charge" in row.iloc[0].lower():
				table_df = table_df.loc[:index - 1]  # Only keep rows before this point
				last_row_value = re.sub('[^0-9.]+', '', str(table_df.iloc[-1, 4]))  # Get the value in column E (index 4) of the last row
				prediction['total'] = round(float(last_row_value), 2)
				break

		if not table_df.empty:
			table_df = table_df.dropna(subset=[table_df.columns[0], table_df.columns[1]])  # Remove NaN values
			table_df = table_df[table_df.iloc[:, 0].str.strip().astype(bool) & table_df.iloc[:, 1].str.strip().astype(bool)] # Remove rows where first or second column is empty or only spaces

		# Set the headers as column names of the table DataFrame
		table_headers = ['product_code', 'goods_description', 'invoice_quantity', 'unit_price', 'price']
		table_df.columns = table_headers
		table_df['unit_quantity'] = ["Piece"] * len(table_df['invoice_quantity'])
		prediction['table'] = table_df.to_dict(orient='list')

	else:
		raise ValueError("The header 'ITEM NO.:' could not be found in the file.")

	return prediction