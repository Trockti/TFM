import pandas as pd
import json
import os

def update_dictionary_file(input_path, json_path, output_path):
    # 1. Detect file type and load data appropriately
    _, ext = os.path.splitext(input_path)
    ext = ext.lower()
    
    print(f"Loading {ext} file...")
    try:
        if ext == '.csv':
            # Try utf-8 first, fallback to latin-1 (common for Spanish text from Excel)
            try:
                df = pd.read_csv(input_path, encoding='utf-8')
            except UnicodeDecodeError:
                df = pd.read_csv(input_path, encoding='latin-1')
        elif ext in ['.xlsx', '.xls']:
            df = pd.read_excel(input_path)
        else:
            print(f"Unsupported file extension: {ext}")
            return
            
        print("Data loaded successfully!")
        
    except Exception as e:
        print(f"Error loading the data file: {e}")
        return

    # 2. Load the JSON dictionary
    try:
        # Try utf-8 first, fallback to latin-1
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                definition_dict = json.load(f)
        except UnicodeDecodeError:
            with open(json_path, 'r', encoding='latin-1') as f:
                definition_dict = json.load(f)
    except Exception as e:
        print(f"Error loading the JSON dictionary: {e}")
        return

    # 3. Check if required columns exist
    if 'Término/expresión' not in df.columns or 'Definición original' not in df.columns:
        print("\nError: Could not find the required columns.")
        print(f"Columns found in your file: {df.columns.tolist()}")
        print("Please check your file and ensure the column names match exactly.")
        return

    # 4. Update definitions
    def update_definition(row):
        term = row['Término/expresión']
        if pd.notna(term) and term in definition_dict:
            return definition_dict[term]
        return row['Definición original']

    df['Definición original'] = df.apply(update_definition, axis=1)

    # 5. Save to the correct format based on the output_path extension
    _, out_ext = os.path.splitext(output_path)
    try:
        if out_ext.lower() == '.csv':
            # utf-8-sig ensures Excel reads the accents correctly when you open the new CSV
            df.to_csv(output_path, index=False, encoding='utf-8-sig') 
        else:
            df.to_excel(output_path, index=False)
            
        print(f"\nSuccess! Update complete. Saved as: {output_path}")
    except Exception as e:
        print(f"Error saving file: {e}")

# --- Example Usage ---
if __name__ == "__main__":
    # Replace these filenames with your actual files
    INPUT_FILE = 'data/Test_set.xlsx'
    JSON_FILE = 'data/cervantes_best.json'
    OUTPUT_FILE = 'Test_set_v2.csv'
    
    update_dictionary_file(INPUT_FILE, JSON_FILE, OUTPUT_FILE)