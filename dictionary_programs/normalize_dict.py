import json
import string

def normalize_key(key):
    """
    Converts a key to lowercase and removes all whitespace and punctuation.
    """
    # Combine all punctuation marks and whitespace characters
    chars_to_remove = string.punctuation + string.whitespace
    
    # Create a translation table to strip these characters
    translator = str.maketrans('', '', chars_to_remove)
    
    # Apply the translation and convert to lowercase
    return str(key).lower().translate(translator)

def normalize_json_keys(data):
    """
    Recursively iterates through lists and dictionaries to normalize keys.
    """
    if isinstance(data, dict):
        normalized_dict = {}
        for key, value in data.items():
            new_key = normalize_key(key)
            # Recursively process the value in case it's a nested dict or list
            normalized_dict[new_key] = normalize_json_keys(value)
        return normalized_dict
    
    elif isinstance(data, list):
        # Recursively process each item in the list
        return [normalize_json_keys(item) for item in data]
    
    else:
        # Base case: if it's just a string, int, bool, etc., return it as is
        return data

def process_json_file(input_filepath, output_filepath):
    """
    Reads a JSON file, normalizes its keys, and writes the output to a new file.
    """
    try:
        # 1. Read the original JSON file
        with open(input_filepath, 'r', encoding='utf-8') as infile:
            data = json.load(infile)

        # 2. Normalize the data
        normalized_data = normalize_json_keys(data)

        # 3. Write the result to a new JSON file
        with open(output_filepath, 'w', encoding='utf-8') as outfile:
            json.dump(normalized_data, outfile, indent=4, ensure_ascii=False)
            
        print(f"Success! Normalized JSON saved to: {output_filepath}")
        
    except FileNotFoundError:
        print(f"Error: The file {input_filepath} was not found.")
    except json.JSONDecodeError:
        print(f"Error: The file {input_filepath} does not contain valid JSON.")

# ==========================================
# Example Usage
# ==========================================
if __name__ == "__main__":
    # Replace these with your actual file paths
    input_file = '../data/definitions_v2.json'   
    output_file = '../data/normalized_definitions.json' 
    
    process_json_file(input_file, output_file)