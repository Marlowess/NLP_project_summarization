import pandas as pd
from transformers import pipeline
import torch

def translation_step(input_filepath, output_filepath, num_of_records, model='Helsinki-NLP/opus-mt-en-it', device=0, batch_size=16):
    """
    This function translates input data from one language to another using GPU and processes in batches.
    """
    # Load the CSV file
    if num_of_records > 0:
        data = pd.read_csv(input_filepath)[:num_of_records]
    else:
        data = pd.read_csv(input_filepath)

    # Check device availability
    if device >= 0:
        if not torch.cuda.is_available():
            print("Warning: GPU is not available. Falling back to CPU.")
            device = -1

    # Initialize the translator
    translator = pipeline("translation", model=model, device=device)

    # Columns to translate
    columns_to_translate = ['paper_title', 'abstract', 'review', 'metareview', 'recommendation']

    # Function to translate a batch of text
    def translate_batch(texts, max_length):
        translations = translator(texts, max_length=max_length, truncation=True)
        return [t['translation_text'] for t in translations]

    # Apply translation to each column in batches
    for column in columns_to_translate:
        if column in data.columns:
            print(f"Translating column: {column}")
            max_len_col = data[column].str.len().max()
            
            # Process in batches
            translated_texts = []
            for i in range(0, len(data), batch_size):
                batch = data[column].iloc[i:i+batch_size].tolist()
                translated_texts.extend(translate_batch(batch, max_length=max_len_col))
            
            data[column] = translated_texts

    # Save the translated file
    data.to_csv(output_filepath, index=False)
    print(f"Translation completed and saved to {output_filepath}")
