from sklearn.model_selection import train_test_split
import pandas as pd
import os

def split_dataset(base_folder_path, input_file_path_list, output_folder_path, random_state=42):
    """
    This function reads the input files and splits the dataset into train, test and validation datasets.
    The datasets are saved in the preprocessed folder.
    """
    # Read all the inputs files and concatenate in a pandas dataframe
    x_label, y_label = ['id', 'review'], 'metareview'
    df = pd.concat([pd.read_csv(f"{base_folder_path}/{file_path}") for file_path in input_file_path_list], ignore_index=True)

    # Split the dataset into train, test and validation, keeping all the columns in the final datasets
    x_train, x_test, y_train, y_test = train_test_split(df[x_label], df[y_label], test_size=0.2, random_state=random_state)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=random_state)

    # Create a complete dataframe for each dataset
    train_df = x_train.copy()
    train_df[y_label] = y_train

    val_df = x_val.copy()
    val_df[y_label] = y_val

    test_df = x_test.copy()
    test_df[y_label] = y_test

    # Save the datasets in the processed folder
    # Check if the output folder exists, otherwise create it
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    train_df.to_csv(f'{output_folder_path}/train.csv', index=False)
    test_df.to_csv(f'{output_folder_path}/test.csv', index=False)
    val_df.to_csv(f'{output_folder_path}/val.csv', index=False)