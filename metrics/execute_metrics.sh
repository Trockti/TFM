#!/bin/bash

# Loop from 1 to 6
for i in {1..6}
do
    INPUT_DIR="../models/results_v${i}"
    OUTPUT_DIR="./results_v${i}"
    LOG_FILE="output_v${i}.log"

    echo "Starting job for version ${i}..."
    
    # Just run the python script normally. It will wait for it to finish.
    python3 hulat_metrics.py --lang es --input_folder "$INPUT_DIR" --out "$OUTPUT_DIR" > "$LOG_FILE" 2>&1
done

echo "All 6 jobs have finished running sequentially!"