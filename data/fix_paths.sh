#!/bin/bash

# Root directory containing the incorrectly named files
root_dir="."

# Find all files matching the pattern with backslashes in the name
find "$root_dir" -maxdepth 1 -type f | while read -r filepath; do
  filename=$(basename "$filepath")
  
  # Check if filename contains backslash characters
  if [[ "$filename" == *\\* ]]; then
    # Replace backslashes with forward slashes to form a path
    newpath=$(echo "$filename" | tr '\\' '/')
    
    # Extract directory path and file name
    dirpath=$(dirname "$newpath")
    basefile=$(basename "$newpath")
    
    # Create the directory path if it doesn't exist
    mkdir -p "$root_dir/$dirpath"
    
    # Move the file to the new directory path
    mv "$root_dir/$filename" "$root_dir/$dirpath/$basefile"
  fi
done

