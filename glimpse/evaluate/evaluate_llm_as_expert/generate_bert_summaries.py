from summarizer import Summarizer
from pathlib import Path
import argparse
import pandas as pd
import os

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summaries", type=Path, default="")
    args = parser.parse_args()
    return args

def bert_summary_f(documents):
    """Summarize using BERT-based extractive summarization."""
    model = Summarizer()
    summaries = [model(doc, num_sentences=3) for doc in documents]
    return " ".join(summaries)

def main():
    args = parse_args()
    # Sample documents
    documents_df = pd.read_csv(args.summaries)
    reviews_by_doc = documents_df.groupby('id')['text'].apply(list).reset_index()
    res_df = pd.DataFrame(columns=['summary', 'reviews'])
    for paper in reviews_by_doc['id']:
        reviews = reviews_by_doc[reviews_by_doc['id'] == paper]['text'].to_numpy()[0]
        bert_summary = bert_summary_f(reviews)
        res_df = pd.concat([res_df, pd.DataFrame({'summary': [bert_summary], 'reviews': [reviews]})], ignore_index=True)
    
    folder_path = "data/evaluation"
    file_name = "bert_summaries.csv"
    file_path = os.path.join(folder_path, file_name)

    os.makedirs(folder_path, exist_ok=True)

    res_df.to_csv(file_path, index=False)
    
if __name__ == "__main__":
    main()