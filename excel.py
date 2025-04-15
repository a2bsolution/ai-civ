import pandas as pd
import numpy as np
import re
from functions import container_separate


def incoterm_filter(incoterm):
	incoterm = re.sub('[^A-Za-z ]+', '', incoterm).upper()
	incoterm_list = ['FOB', 'CFR', 'CIF', 'CIP', 'CPT', 'DAP', 'DAT', 'DDP', 'DPU', 'EXW', 'FAS', 'FC1', 'FC2', 'FCA']

	for word in incoterm_list:
		if word in incoterm:
			return word
			break
	return None

def extract_xlsx(df):
	prediction = {'page': [1], 'invoice_number': None, 'incoterm': None, 'container': None}
	prediction['supplier'] = next((val for row in df.itertuples(index=False) for val in row if pd.notna(val)), None)

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

				elif "container" in value_clean:
					if col_idx + 1 < df.shape[1] and pd.notna(df.iloc[row_idx, col_idx + 1]) and str(df.iloc[row_idx, col_idx + 1]).strip():
						prediction['container'] = container_separate(df.iloc[row_idx, col_idx + 1])[0]

			# If both values are found, exit the loop early
		if prediction['invoice_number'] is not None and prediction['incoterm'] is not None and prediction['container'] is not None:
			break

	item_no_row_index = df[df.apply(lambda row: row.astype(str).str.contains("ITEM NO", case=False, na=False).any(), axis=1)].index
	if not item_no_row_index.empty:
		header_row = item_no_row_index[0]
		table_headers = df.iloc[header_row]  # Grab the headers from row 20
		table_df = df.iloc[header_row + 1:]  # Extract the data from the rows below the headers

		normalized_headers = table_headers.astype(str).str.strip().str.lower()
		matching_columns = normalized_headers[normalized_headers.str.contains("total gross weight", na=False)]
		total_weight_col_idx = matching_columns.index[0] if not matching_columns.empty else -1

		#try:
		prediction['currency'] = df.iloc[header_row, 4]
		table_df = table_df.iloc[:, :5] 
		table_df = table_df.dropna(how='all')  # Drop rows that are entirely empty

		# Stop reading the table when "Inland Charge USD" or "Total" is encountered in the first column
		for index, row in reversed(list(table_df.iterrows())):
			if isinstance(row.iloc[0], str) and "inland charge" in row.iloc[0].lower(): #for some reason row points to row after inland charge
				table_df = table_df.loc[:index - 2]  # Only keep rows before this point, and drop the last row
				last_row_value = re.sub('[^0-9.]+', '', str(table_df.iloc[-1, 4]))  # Get the value in column E (index 4) of the last row
				prediction['total'] = round(float(last_row_value), 2)

				if total_weight_col_idx != -1:
					total_weight_value = df.iloc[index-2, total_weight_col_idx]
					cleaned_weight = re.sub(r'[^0-9.]+', '', str(total_weight_value))
					prediction['total_weight'] = round(float(cleaned_weight), 2) if cleaned_weight else 0.0

				break
			elif isinstance(row.iloc[0], str) and "total" in row.iloc[0].lower(): #row points to total row
				table_df = table_df.loc[:index - 1]
				prediction['total'] = round(float(row.iloc[4]), 2)

				if total_weight_col_idx != -1:
					total_weight_value = df.iloc[index,total_weight_col_idx]
					cleaned_weight = re.sub(r'[^0-9.]+', '', str(total_weight_value))
					prediction['total_weight'] = round(float(cleaned_weight), 2) if cleaned_weight else 0.0

				break

		if not table_df.empty:
			table_df = table_df[~table_df.iloc[:, 0].astype(str).str.contains("total", case=False, na=False)]
			table_df = table_df.dropna(subset=[table_df.columns[0], table_df.columns[1]])  # Remove NaN values
			table_df = table_df[table_df.iloc[:, 0].str.strip().astype(bool) & table_df.iloc[:, 1].str.strip().astype(bool)] # Remove rows where first or second column is empty or only spaces

		# Set the headers as column names of the table DataFrame
		#table_df = table_df.iloc[:, :5] 
		table_headers = ['product_code', 'goods_description', 'invoice_quantity', 'unit_price', 'price']
		table_df.columns = table_headers
		table_df['unit_quantity'] = ["Piece"] * len(table_df['invoice_quantity'])

		prediction['table'] = table_df.to_dict(orient='list')
		#except:
			#return prediction

	return prediction

def extract_xls(df):
	prediction = {'page': [1], 'invoice_number': None, 'incoterm': None, 'container': None}

	prediction['supplier'] = next((val for row in df.itertuples(index=False) for val in row if pd.notna(val)), None)

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

				elif "container" in value_clean:
					if col_idx + 1 < df.shape[1] and pd.notna(df.iloc[row_idx, col_idx + 1]) and str(df.iloc[row_idx, col_idx + 1]).strip():
						prediction['container'] = container_separate(df.iloc[row_idx, col_idx + 1])[0]

			# If both values are found, exit the loop early
		if prediction['invoice_number'] is not None and prediction['incoterm'] is not None and prediction['container'] is not None:
			break

	item_no_positions = df.apply(lambda row: row.astype(str).str.contains("ITEM NO", case=False, na=False), axis=1)
	first_item_no_position = item_no_positions.stack()
	header_row, header_col = first_item_no_position[first_item_no_position].index[0]
	
	if not first_item_no_position.empty:
		table_headers = df.iloc[header_row]
		table_df = df.iloc[header_row + 1:]  # Extract the data from the rows below the headers

		normalized_headers = table_headers.astype(str).str.strip().str.lower()
		matching_columns = normalized_headers[normalized_headers.str.contains("total gross weight", na=False)]
		total_weight_col_idx = matching_columns.index[0] if not matching_columns.empty else -1

		prediction['currency'] = df.iloc[header_row, 4]

		table_df = table_df.iloc[:, header_col:header_col+5]
		table_df = table_df.dropna(how='all')  # Drop rows that are entirely empty

		for index, row in table_df.iterrows():
			if isinstance(row.iloc[0], str) and "inland charge" in row.iloc[0].lower():
				table_df = table_df.loc[:index - 1]  # Only keep rows before this point
				last_row_value = re.sub('[^0-9.]+', '', str(table_df.iloc[-1, 4]))  # Get the value in column E (index 4) of the last row
				prediction['total'] = round(float(last_row_value), 2)

				if total_weight_col_idx != -1:
						total_weight_value = df.iloc[index-2, total_weight_col_idx]
						prediction['total_weight'] = round(float(re.sub(r'[^0-9.]+', '', str(total_weight_value))), 2)

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