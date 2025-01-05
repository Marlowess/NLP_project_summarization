import pandas as pd
from transformers import pipeline
import torch

def translation_step(input_filepath, output_filepath, num_of_records, model='Helsinki-NLP/opus-mt-en-it', device=-1):
    """
    This function translates input data from a language to another one
    """
    # Load the CSV file
    if num_of_records > 0:
        data = pd.read_csv(input_filepath)[:num_of_records]
    else:
        data = pd.read_csv(input_filepath)

    # Initialize the translator
    translator = pipeline("translation", model=model, device=device)

    # Columns to translate
    columns_to_translate = ['paper_title', 'abstract', 'review', 'metareview', 'recommendation']

    # Function to translate a column in the dataframe
    def translate_column(data, column, max_length):
        return data[column].apply(lambda x: translator(x, max_length=max_length)[0]['translation_text'] if isinstance(x, str) else x)

    # Apply translation to each column
    for column in columns_to_translate:
        max_len_col = data[column].str.len().max()
        if column in data.columns:
            data[column] = translate_column(data, column, max_len_col)

    # Save the translated file
    data.to_csv(output_filepath, index=False)

    print(f"Result saved in: {output_filepath}")
