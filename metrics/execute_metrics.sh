#!/bin/bash

# Loop from 1 to 6
for i in {1..6}
do
    INPUT_DIR="../models/results_v${i}"
    OUTPUT_DIR="./results_v${i}"
    LOG_FILE="output_v${i}.log"

    echo "Starting job for version ${i}..."
    
    nohup python3 hulat_metrics.py --lang es --input_folder "$INPUT_DIR" --out "$OUTPUT_DIR" > "$LOG_FILE" 2>&1
done

echo "All 6 jobs have been successfully started in the background!"
echo "You can check their progress by reading the output_v*.log files."