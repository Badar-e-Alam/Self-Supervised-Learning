#!/bin/bash

# Change directory to /scratch
cd /scratch

# Make a new directory named mrvl005h
mkdir mrvl005h

# Change directory to /scratch/mrvl005h
cd mrvl005h

# Copy the Image_data.zip file from $HPCVAULT/data to the current directory with progress
rsync --progress $HPCVAULT/data/Image_data.zip .

# Unzip the Image_data.zip file
total_size=$(unzip -l Image_data.zip | awk 'END {print $1}')
unzip Image_data.zip 
rm Image_data.zip