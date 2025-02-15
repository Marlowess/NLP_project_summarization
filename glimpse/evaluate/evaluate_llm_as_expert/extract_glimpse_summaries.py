import pandas as pd
import argparse
from pathlib import Path
import os

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rsa_res", type=Path, default="", required=True)
    parser.add_argument("--reviews", type=Path, default="", required=True)
    args = parser.parse_args()
    return args


def get_reviews_by_id(reviews_df, paper_id):
    grouped_reviews = reviews_df.groupby('id')['text'].apply(list).reset_index()
    reviews_by_id = grouped_reviews[grouped_reviews['id'] == paper_id]['text'].values[0]
    return reviews_by_id

def make_summaries_by_reviews(rsa_res_df, reviews_df):
    glimpse_unique_data = []
    glimpse_speaker_data = []
    for _, row in rsa_res_df.iterrows():
        paper_id = row['id'][0]
        reviews_by_id = get_reviews_by_id(reviews_df, paper_id)
        consensus_samples = row['consensuality_scores'].sort_values(ascending=True).head(3).index.tolist()
        dissensus_samples = row['consensuality_scores'].sort_values(ascending=False).head(3).index.tolist()
        rsa_samples = row['best_rsa'].tolist()[:3]
        rsa = ".".join(rsa_samples)
        consensus = ".".join(consensus_samples)
        dissensus = ".".join(dissensus_samples)
        glimpse_speaker_summary = consensus + "\n\n" + rsa
        glimpse_unique_summary = consensus + "\n\n" + dissensus
        glimpse_speaker_row = {'id': paper_id, 'summary': glimpse_speaker_summary, 'reviews': reviews_by_id}
        glimpse_unique_row = {'id': paper_id,'summary': glimpse_unique_summary, 'reviews': reviews_by_id}
        glimpse_speaker_data.append(glimpse_speaker_row)
        glimpse_unique_data.append(glimpse_unique_row)
    
    glimpse_speaker_summaries_by_reviews_df = pd.DataFrame(glimpse_speaker_data)
    glimpse_unique_summaries_by_reviews_df = pd.DataFrame(glimpse_unique_data)
    return glimpse_unique_summaries_by_reviews_df, glimpse_speaker_summaries_by_reviews_df

def main():
    args = parse_args()
    isExtractive = "extractive" in str(args.rsa_res)
    prefix = "extractive" if isExtractive else "abstractive"
    rsa_res_df = pd.read_pickle(args.rsa_res)
    rsa_res_df = pd.DataFrame(rsa_res_df['results'])
    reviews_df = pd.read_csv(args.reviews)
    (
        glimpse_unique_summaries_by_reviews_df, 
        glimpse_speaker_summaries_by_reviews_df
    ) = make_summaries_by_reviews(rsa_res_df, reviews_df)

    folder_path = "data/evaluation"
    glimpse_unique_file_name = f"{prefix}_glimpse_unique_summaries.json"
    glimpse_speaker_file_name = f"{prefix}_glimpse_speaker_summaries.json"
    glimpse_unique_file_path = os.path.join(folder_path, glimpse_unique_file_name)
    glimpse_speaker_file_path = os.path.join(folder_path, glimpse_speaker_file_name)

    os.makedirs(folder_path, exist_ok=True)

    glimpse_unique_summaries_by_reviews_df.to_json(glimpse_unique_file_path, index=False)
    glimpse_unique_summaries_by_reviews_df.to_csv(f"data/evaluation/{prefix}_glimpse_unique_summaries.csv", index=False)
    glimpse_speaker_summaries_by_reviews_df.to_json(glimpse_speaker_file_path, index=False)
    glimpse_speaker_summaries_by_reviews_df.to_csv(f"data/evaluation/{prefix}_glimpse_speaker_summaries.csv", index=False)

if __name__ == "__main__":
    main()