import pandas as pd
import argparse
from pathlib import Path
import os

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summaries", type=Path, default="", required=True)
    parser.add_argument("--reviews", type=Path, default="", required=True)
    parser.add_argument("--base_path", type=str, default="", required=True)
    parser.add_argument("--output_path", type=str, default="", required=True)
    args = parser.parse_args()
    return args


def get_reviews_by_id(reviews_df, paper_id):
    grouped_reviews = reviews_df.groupby('id')['text'].apply(list).reset_index()
    reviews_by_id = grouped_reviews[grouped_reviews['id'] == paper_id]['text'].values[0]
    return reviews_by_id

def make_summaries_by_reviews(summaries_df, reviews_df):
    summaries_by_review_data = []
    for _, row in summaries_df.iterrows():
        paper_id = row['id']
        summary = row['summary']
        reviews_by_id = get_reviews_by_id(reviews_df, paper_id)
        row = {'id': paper_id, 'summary': summary, 'reviews': reviews_by_id}
        summaries_by_review_data.append(row)
    
    summaries_by_reviews_df = pd.DataFrame(summaries_by_review_data)
    return summaries_by_reviews_df

def main():
    args = parse_args()
    summaries_df = pd.read_csv(args.summaries)
    model = summaries_df['metadata/method'][0]
    reviews_df = pd.read_csv(args.reviews)
    summaries_by_reviews_df = make_summaries_by_reviews(summaries_df, reviews_df)

    folder_path = f"{args.output_path}/data/evaluation"
    file_name = f"{model}.json"
    file_path = os.path.join(folder_path, file_name)

    os.makedirs(folder_path, exist_ok=True)

    summaries_by_reviews_df.to_json(file_path, index=False)
    summaries_by_reviews_df.to_csv(f"{args.output_path}/data/evaluation/{model}.csv", index=False)

if __name__ == "__main__":
    main()