import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import os
import matplotlib.pyplot as plt

def split_data(train_df, unique_counts, train_ratio=0.9):
    train_data = []
    val_data = []

    for coil_id, count in unique_counts.items():
        # Get the samples for the current coil_id
        samples = train_df[train_df['xxx'] == coil_id]

        # Calculate the number of samples for training
        train_samples = int(count * train_ratio)

        # Select a random subset of samples for training
        train_samples = samples.sample(n=train_samples, random_state=1)

        # The validation samples are the ones not included in the training samples
        val_samples = samples.drop(train_samples.index)

        # Append the train and validation samples to the respective lists
        train_data.append(train_samples)
        val_data.append(val_samples)
    # Concatenate the train and validation data
    train_data = pd.concat(train_data)
    val_data = pd.concat(val_data)

    # Calculate the counts for each category in the train and validation data
    # train_counts = train_data['xxx'].value_counts()
    # val_counts = val_data['xxx'].value_counts()

    # Write train_counts and val_counts to CSV files
    # for df in train_data:
    #     df.drop(columns=['xxx'])
    #     df.reset_index(drop=True)
    train_data.to_csv('train.csv', sep=',')
    # for ata in val_data:
    #     ata.drop(columns=['xxx'])

    #     ata.reset_index(drop=True, inplace=True)
    val_data.to_csv('validation.csv', sep=',')

    # return train_data, val_data
    
    # # Plot the counts
    # # plt.bar(val_counts.index, val_counts.values, label='Validation')
    # # plt.bar(train_counts.index, train_counts.values, bottom=val_counts[train_counts.index].values, label='Train')
    # # plt.xlabel('Coil ID')
    # # plt.ylabel('Image Count')
    # # plt.title('Train-Validation Split')
    # # plt.legend()
    # # plt.xticks(rotation=90)
    # # plt.show()
    # return train_data, val_data
def clean():
    import csv

    # Open the CSV file
    with open('train_val.csv', 'r') as f:
        reader = csv.reader(f)
        data = list(reader)

    # Remove the double quotes from the start and end of each line
    cleaned_data = []
    for row in data:
        # Split the string into separate fields
        cleaned_row = row[0].split(',')
        cleaned_data.append(cleaned_row)

    # Write the cleaned data back to the CSV file
    with open('saaf.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(cleaned_data)

def plot_unique_counts(train_df):
    # Extract the xxx values from the image column
    train_df['xxx'] = train_df['image'].str.extract(r'Spule(\w+)_Image')  # Adjusted the regex pattern

    # Count the unique xxx values
    unique_counts = train_df['xxx'].value_counts()

    # Create a histogram based on the unique counts
    plt.bar(unique_counts.index, unique_counts.values)
    plt.xlabel('Coil ID')
    plt.ylabel('Image Count')
    plt.title('Histogram of Unique Coil Values')
    plt.xticks(rotation=90)
    plt.show()

    return unique_counts


if __name__=="__main__":
    train_df = pd.read_csv("linear_winding_labels_image_level_train_val_set_v2.0.csv", sep=",")
    test_df = pd.read_csv("linear_winding_labels_image_level_test_set_v2.0.csv", sep=",")
    unique_counts_train = plot_unique_counts(train_df)
    unique_counts_test = plot_unique_counts(test_df)
    split_data(train_df, unique_counts_train)
    import pdb;pdb.set_trace()
