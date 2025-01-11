from summarizer import Summarizer
from pathlib import Path
import argparse
import pandas as pd

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
    for paper in reviews_by_doc['id']:
        documents = reviews_by_doc[reviews_by_doc['id'] == paper]['text'].to_numpy()[0]
        bert_summary = bert_summary_f(documents)
        # todo save summaries
    
if __name__ == "__main__":
    main()