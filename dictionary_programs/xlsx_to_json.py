import pandas as pd
import json

def transform_excel_to_json(excel_filename, json_filename):
    # 1. Load the data
    try:
        df = pd.read_excel(excel_filename)
        
        # DEBUG: Print actual columns found
        print("🔍 Columns found in your Excel file:")
        print(df.columns.tolist())
        print("-" * 50)
        
    except FileNotFoundError:
        print(f"Error: The file '{excel_filename}' was not found.")
        return
    except Exception as e:
        print(f"An error occurred: {e}")
        return

    # ==========================================
    # ⚙️ CONFIGURATION: Map your Excel columns here
    # ==========================================
    EXCEL_COL_ID_TERM = "ID_Término"      
    EXCEL_COL_TERM = "Término/expresión"
    EXCEL_COL_ORIGINAL = "Definición original"
    
    EXCEL_COL_ADAPTED = "Definición adaptada"
    EXCEL_COL_TRANSLATION = "Traducción / Sinónimos"
    EXCEL_COL_CONTEXT = "Frase_ejemplo"
    # Corresponding ID columns
    EXCEL_COL_DEF_ID = "ID_Definición"
    EXCEL_COL_AD_ID = "ID_Definición_adaptada"
    EXCEL_COL_SYN_ID = "ID_Sinónimo"
    # ==========================================

    json_data = []

    for index, row in df.iterrows():
        def is_valid(val):
            return pd.notna(val) and str(val).strip() != ""

        adapted_val = None
        target_id_val = None

        if EXCEL_COL_ADAPTED in row and is_valid(row[EXCEL_COL_ADAPTED]):
            adapted_val = str(row[EXCEL_COL_ADAPTED]).strip()
            target_id_val = row.get(EXCEL_COL_AD_ID)
            
        elif EXCEL_COL_TRANSLATION in row and is_valid(row[EXCEL_COL_TRANSLATION]):
            adapted_val = str(row[EXCEL_COL_TRANSLATION]).strip()
            target_id_val = row.get(EXCEL_COL_SYN_ID)

        term_val = row.get(EXCEL_COL_ID_TERM)
        id_definition_val = row.get(EXCEL_COL_DEF_ID)
        new_key = f"{term_val}_{index}"
        # Build the JSON object
        term_obj = {
            (term_val + "_" + id_definition_val): {
                "id_term": term_val,
                "term": row.get(EXCEL_COL_TERM),
                "original": row.get(EXCEL_COL_ORIGINAL),
                "adapted": adapted_val,
                "id_definition": id_definition_val,
                "id_adaptation": target_id_val,
                "context": row.get(EXCEL_COL_CONTEXT)
            }
        }

        # Clean up NaN/Null values for cleaner JSON
        for key, value in term_obj.items():
            if pd.isna(value):
                term_obj[key] = None

        json_data.append(term_obj)

    with open(json_filename, 'w', encoding='utf-8') as json_file:
        json.dump(json_data, json_file, indent=4, ensure_ascii=False)
        
    print(f"✅ Successfully transformed {len(json_data)} records and saved to '{json_filename}'")

if __name__ == "__main__":
    input_file = "../data/Test_set_v3.xlsx"
    output_file = "../data/transformed_terms.json"
    
    transform_excel_to_json(input_file, output_file)