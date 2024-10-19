#!/bin/bash

# Path to the text file containing the list of .off files
FILE_LIST="path/to/your/file_list.txt"
OUTPUT_PATH="path/to/your/output/directory"
KIND_OF_OUTPUTS='{"critical and non-critical points": true, "only critical points": false, "objet": true}'
AFFICHAGE="true"  # or "false" depending on your needs

# Check if the file list exists
if [ ! -f "$FILE_LIST" ]; then
    echo "File list not found: $FILE_LIST"
    exit 1
fi

# Read the file line by line
while IFS= read -r input_path; do
    # Check if the line is not empty
    if [ -n "$input_path" ]; then
        # Run the Python script for each .off file
        python /home/pelissier/These-ATER/Papier_international3/PointNet/my_PointNet/get_critical_points.py "$input_path" "$OUTPUT_PATH" --kind_of_outputs "$KIND_OF_OUTPUTS" --affichage "$AFFICHAGE"
    fi
done < "$FILE_LIST"
