#!/bin/bash

# Check if file2.bed exists
if [ ! -f "hg38_test_sequences.bed" ]; then
  echo "Error: file2.bed does not exist in the current directory."
  exit 1
fi

# Loop through each .bed file in the current directory
for bed_file in *.bed; do
  # Skip file2.bed to avoid self-intersection
  if [ "$bed_file" == "hg38_test_sequences.bed" ]; then
    continue
  fi

  # Define the output file name
  output_file="${bed_file%.bed}_intersected.bed"

  # Perform the intersection
  bedtools intersect -a "$bed_file" -b hg38_test_sequences.bed -wa | sort -k1,1 -k2,2n > "$output_file"

  # Notify the user
  echo "Intersection of $bed_file and file2.bed written to $output_file"
done

echo "All intersections completed!"
