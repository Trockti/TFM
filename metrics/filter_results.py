import json
import csv
from collections import defaultdict

input_filename = 'text_analysis_results.json'
json_output_filename = 'compliance_summary.json'
csv_output_filename = 'compliance_summary.csv'

with open(input_filename, 'r', encoding='utf-8') as f:
    data = json.load(f)

# Dictionary to hold the aggregated counts and scores
# The Key will be a tuple: (Folder, Model)
summary_data = defaultdict(lambda: {
    "Total_Items": 0,
    "Sum_Compliance_Score": 0.0,
    "Count_Max_25_Words": 0,
    "Count_Max_2_Sentences": 0,
    "Count_Short_Sentences": 0,
    "Count_No_Double_Negation": 0,
    "Count_No_Self_Definition": 0,
    "Count_Active_Voice_Proxy": 0,
    "Count_Category_Function": 0
})

for item in data:
    folder = item.get("Folder", "Unknown")
    model = item.get("Model", "Unknown")
    key = (folder, model)
    
    # 1. Increment total items and sum the compliance score for the mean calculation
    summary_data[key]["Total_Items"] += 1
    summary_data[key]["Sum_Compliance_Score"] += item.get("Prompt Compliance Score", 0.0)
    
    # 2. Increment specific compliance counters if they are True
    if item.get("Compliance: Max 25 Words") is True:
        summary_data[key]["Count_Max_25_Words"] += 1
        
    if item.get("Compliance: Max 2 Sentences") is True:
        summary_data[key]["Count_Max_2_Sentences"] += 1
        
    if item.get("Compliance: Short Sentences (<15 words/sent)") is True:
        summary_data[key]["Count_Short_Sentences"] += 1
        
    if item.get("Compliance: No Double Negation") is True:
        summary_data[key]["Count_No_Double_Negation"] += 1
        
    if item.get("Compliance: No Self-Definition") is True:
        summary_data[key]["Count_No_Self_Definition"] += 1
        
    if item.get("Compliance: Active Voice Proxy") is True:
        summary_data[key]["Count_Active_Voice_Proxy"] += 1
        
    if item.get("Compliance: Category+Function Pattern") is True:
        summary_data[key]["Count_Category_Function"] += 1

# Flatten the dictionary, calculate the mean, and format for CSV and JSON output
final_results = []
for (folder, model), stats in summary_data.items():
    total_items = stats["Total_Items"]
    mean_score = stats["Sum_Compliance_Score"] / total_items if total_items > 0 else 0
    
    row = {
        "Folder": folder,
        "Model": model,
        "Mean_Compliance_Score": round(mean_score, 4), # Rounded to 4 decimal places
        "Max_25_Words": stats["Count_Max_25_Words"],
        "Max_2_Sentences": stats["Count_Max_2_Sentences"],
        "Short_Sentences": stats["Count_Short_Sentences"],
        "No_Double_Negation": stats["Count_No_Double_Negation"],
        "No_Self_Definition": stats["Count_No_Self_Definition"],
        "Active_Voice_Proxy": stats["Count_Active_Voice_Proxy"],
        "Category_Function": stats["Count_Category_Function"]
    }
    final_results.append(row)

# Save to JSON
with open(json_output_filename, 'w', encoding='utf-8') as f:
    json.dump(final_results, f, indent=4)

# Save to CSV
if final_results:
    with open(csv_output_filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=final_results[0].keys())
        writer.writeheader()
        writer.writerows(final_results)

print("Summary processing complete. Files saved.")