import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import os
import matplotlib.pyplot as plt
import sys
import random
sys.path.append("..")
import seaborn as sns
def split_train_validation(train_df, unique_counts, train_ratio=0.9):
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
    train_counts = train_data['xxx'].value_counts()
    val_counts = val_data['xxx'].value_counts()

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

    #Plot the counts
    plt.bar(val_counts.index, val_counts.values, label='Validation')
    plt.bar(train_counts.index, train_counts.values, bottom=val_counts[train_counts.index].values, label='Train')
    plt.xlabel('Coil ID')
    plt.ylabel('Image Count')
    plt.title('Train-Validation Split')
    plt.legend()
    plt.xticks(rotation=90)
    plt.show()
    return train_data, val_data


def out_dis_split(dataframe, val_coil=6):
    # Extract the coils numbers from the "image" column
    dataframe['CoilNumber'] = dataframe['image'].str.extract('(\d+)').astype(
        int)
    # Count the unique coil numbers
    unique_counts = dataframe['CoilNumber'].unique()

    #check if there are engough uniques coils for splitting
    if len(unique_counts) < val_coil:
        print("There are not enough unique coils for splitting")
        return None
    val_coils = random.sample(list(unique_counts), val_coil)
    train_coils = list(set(unique_counts) - set(val_coils))
    # Split the data into train and validation sets
    val_data = dataframe[dataframe['CoilNumber'].isin(val_coils)]
    train_data = dataframe[dataframe['CoilNumber'].isin(train_coils)]
    # Plot the counts
    plt.bar(train_coils,
            train_data['CoilNumber'].value_counts().values,
            label='Train')
    plt.bar(val_coils,
            val_data['CoilNumber'].value_counts().values,
            label='Validation')
    plt.xlabel('Coil ID')
    plt.ylabel('Image Count')
    plt.title('Train-Validation Split')
    plt.legend()
    plt.xticks(rotation=90)
    plt.show()

def balanced_split(dataframe, val_coil=5):
    # Extract the coils numbers from the "image" column
    dataframe['CoilNumber'] = dataframe['image'].str.extract('(\d+)').astype(int)

    # Count the unique coil numbers
    unique_counts = dataframe['CoilNumber'].unique()

    # Check if there are enough unique coils for splitting
    if len(unique_counts) < val_coil:
        print("There are not enough unique coils for splitting")
        return None

    # Keep generating random coils until it finds ones that have an equal number of 0s and 1s in the "binary_NOK" column
    while True:
        val_coils = np.random.choice(unique_counts, val_coil, replace=False)
        val_data = dataframe[dataframe['CoilNumber'].isin(val_coils)]
        binary_NOK_counts = val_data['binary_NOK'].value_counts()
        if len(binary_NOK_counts) == 2 and binary_NOK_counts[0] == binary_NOK_counts[1]:
            break

    train_coils = list(set(unique_counts) - set(val_coils))
    train_data = dataframe[dataframe['CoilNumber'].isin(train_coils)]

    # Plot the counts
    plt.bar(train_coils, train_data['CoilNumber'].value_counts().values, label='Train')
    plt.bar(val_coils, val_data['CoilNumber'].value_counts().values, label='Validation')
    plt.xlabel('Coil ID')
    plt.ylabel('Image Count')
    plt.title('Train-Validation Split')
    plt.legend()
    plt.xticks(rotation=90)
    plt.show()

    return train_data, val_data
    # train_dis = train_data["binary_NOK"]
    # val_dis = val_data["binary_NOK"]
    # plt.hist(train_dis, bins=2, label="train")
    # plt.hist(val_dis, bins=2, label="val")
    # plt.legend("Binary NOK")
    # plt.show()
    return train_data, val_data


def data_split_test_train_validation(df, unique_counts, train_ratio=0.7, val_ratio=0.2):
    train_data = []
    val_data = []
    test_data = []

    for coil_id, count in unique_counts.items():
        # Get the samples for the current coil_id
        samples = df[df['xxx'] == coil_id]

        # Calculate the number of samples for training and validation
        train_samples_count = int(count * train_ratio)
        val_samples_count = int(count * val_ratio)

        # Select a random subset of samples for training
        train_samples = samples.sample(n=train_samples_count, random_state=1)

        # Drop the training samples to get the remaining samples
        remaining_samples = samples.drop(train_samples.index)

        # Select a random subset of remaining samples for validation
        val_samples = remaining_samples.sample(n=val_samples_count, random_state=1)

        # The test samples are the ones not included in the training and validation samples
        test_samples = remaining_samples.drop(val_samples.index)

        # Append the train, validation and test samples to the respective lists
        train_data.append(train_samples)
        val_data.append(val_samples)
        test_data.append(test_samples)

    # Concatenate the train, validation and test data
    train_data = pd.concat(train_data)
    val_data = pd.concat(val_data)
    test_data = pd.concat(test_data)

    # Calculate the counts for each category in the train, validation and test data
    train_counts = train_data['xxx'].value_counts()
    val_counts = val_data['xxx'].value_counts()
    test_counts = test_data['xxx'].value_counts()

    # Write train_data, val_data and test_data to CSV files
    train_data.to_csv('train.csv', sep=',')
    val_data.to_csv('validation.csv', sep=',')
    test_data.to_csv('test.csv', sep=',')

    # Plot the counts
    plt.bar(train_counts.index, train_counts.values, label='Train')
    plt.bar(val_counts.index, val_counts.values, bottom=train_counts[val_counts.index].values, label='Validation')
    plt.bar(test_counts.index, test_counts.values, bottom=(train_counts[test_counts.index].values + val_counts[test_counts.index].values), label='Test')
    plt.xlabel('Coil ID')
    plt.ylabel('Image Count')
    plt.title('Train-Validation-Test Split')
    plt.legend()
    plt.xticks(rotation=90)
    plt.show()

    return train_data, val_data, test_data
def clean(path="../data_csv/train_val.csv"):
    import csv

    # Open the CSV file
    with open(path, 'r') as f:
        reader = csv.reader(f)
        data = list(reader)

    # Remove the double quotes from the start and end of each line
    cleaned_data = []
    for row in data:
        # Split the string into separate fields
        cleaned_row = row[0].split(',')
        cleaned_data.append(cleaned_row)

    # Write the cleaned data back to the CSV file
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(cleaned_data)

def plot_unique_counts(train_df):
    # Extract the xxx values from the image column
    train_df['xxx'] = train_df['image'].str.extract(r'Spule(\w+)_Image')  # Adjusted the regex pattern

    # Count the unique xxx values
    unique_counts = train_df["xxx"].value_counts()

    #Create a histogram based on the unique counts
    plt.bar(unique_counts.index, unique_counts.values)
    plt.xlabel('Coil ID')
    plt.ylabel('Image Count')
    plt.title('Histogram of Unique Coil IDs in')
    plt.xticks(rotation=90)
    plt.show()

    return unique_counts
def label_dis_sns(df):
    columns = ['binary_NOK', 'multi-label_double_winding', 'multi-label_gap', 'multi-label_crossing', 'CoilNumber']

    # Create a new figure
    fig = plt.figure(figsize=(15, 10))

    # Iterate over the column names and create a countplot for each
    for i, column in enumerate(columns, start=1):
        if column in df.columns:
            # Create a subplot
            ax = fig.add_subplot(len(columns), 1, i)

            # Create a countplot
            sns.countplot(x=column, data=df, ax=ax)

            # Set the title
            ax.set_title('Distribution of ' + column)

    # Adjust the subplot parameters
    plt.tight_layout()

    # Display the plot
    plt.show()



if __name__ == "__main__":
    #clean("../data_csv\linear_winding_labels_image_level_train_val_set_v2.0.csv")
    original_test = pd.read_csv("../data_csv\linear_winding_labels_image_level_test_set_v2.0.csv", sep=",")
    original_train = pd.read_csv("../data_csv\linear_winding_labels_image_level_train_val_set_v2.0.csv", sep=",")
    train,val=balanced_split(original_train)
    unique_counts = label_dis_sns(original_train)
    import pdb
    pdb.set_trace()
    # unique_train = pd.read_csv("unique_train.csv", sep=",")
    # unique_val = pd.read_csv("unique_validation.csv", sep=",")
    # unique_test = pd.read_csv("../data_csv/test.csv", sep=",")
    # unique_counts_train = plot_unique_counts(unique_train)
    # unique_counts_val = plot_unique_counts(unique_val)
    # unique_counts_test = plot_unique_counts(unique_test)
    # label_dis_sns(unique_train)
    # label_dis_sns(unique_val)
    # label_dis_sns(unique_test)

    # train_df = pd.read_csv("../data_csv/train_val.csv", sep=",")
    # train,val=out_dis_split(train_df)
    # import pdb
    # pdb.set_trace()
    # unique_coils = plot_unique_counts(train_df=train_df)
    # # plot_unique_counts(val_)
    # # plot_unique_counts(test_)

    # # test_df = pd.read_csv("linear_winding_labels_image_level_test_set_v2.0.csv", sep=",")
    # unique_counts_train = plot_unique_counts(train_df)
    # # unique_counts_test = plot_unique_counts(test_df)
    # split_train_validation(train_df, unique_counts_train)
    # import pdb
    # pdb.set_trace()
