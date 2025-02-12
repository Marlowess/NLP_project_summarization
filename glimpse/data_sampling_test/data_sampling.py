import pandas as pd
import argparse
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("file_path", type=str)
    #parser.add_argument("output_file", type=str, default="samples.csv", required=False)
    parser.add_argument("sample_fraction", type=float)
    args = parser.parse_args()

    try:
        data = pd.read_csv(args.file_path)
    except FileNotFoundError:
        print(f"Error: file '{args.file_path}' not found.")
        return

    if not 0 < args.sample_fraction <= 1:
        print("Error: value must be between 0 and 1")
        return
    
    output_dir = os.path.dirname('data/sample/samples.csv')
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Directory '{output_dir}' created.")

    # sample by id (e.g. if i got 100 id i want % of id and all the reviews for those id)
    unique_ids = data['id'].unique()
    sampled_ids = pd.Series(unique_ids).sample(frac=args.sample_fraction, random_state=42)
    sampled_data = data[data['id'].isin(sampled_ids)]

    sampled_data.to_csv('data/sample/samples.csv', index=False)
    print(f"Sampling completed. File saved as samples.csv.")

if __name__ == "__main__":
    main()
