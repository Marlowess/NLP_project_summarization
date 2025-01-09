import os
from pathlib import Path
import argparse
from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from summarizer import Summarizer
from lexrank import LexRank
from lexrank.mappings.stopwords import STOPWORDS
from nltk.tokenize import sent_tokenize
import numpy as np
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summaries", type=Path, default="")
    args = parser.parse_args()
    return args

def lexrank_summary(documents):
    """Summarize using LexRank."""
    sentences = []
    for doc in documents:
        sentences.extend(sent_tokenize(doc))
    lexrank = LexRank(sentences, stopwords=STOPWORDS['en'])
    return lexrank.get_summary(sentences, summary_size=3)

def lsa_summary(documents):
    """Summarize using Latent Semantic Analysis."""
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(documents)
    svd = TruncatedSVD(n_components=1)
    svd.fit(tfidf_matrix)
    components = svd.components_.flatten()
    summary_idx = components.argsort()[-3:][::-1]
    return [documents[i] for i in summary_idx]

def bert_summary(documents):
    """Summarize using BERT-based extractive summarization."""
    model = Summarizer()
    summaries = [model(doc, num_sentences=3) for doc in documents]
    return " ".join(summaries)

def llama_summary(documents):
    """Summarize using Llama-7B instruct."""
    summarizer = pipeline('summarization', model='openai/llama-7b-instruct')
    combined_text = " ".join(documents)
    return summarizer(combined_text, max_length=200, min_length=50, do_sample=False)[0]['summary_text']

def evaluate_summaries(documents, summaries):
    """Evaluate summaries using a large language model."""
    evaluator = pipeline('text-generation', model='gpt-4')  # Assuming GPT-4 API or similar
    evaluation_prompts = []

    for i, summary in enumerate(summaries):
        prompt = f"Given the following documents:\n{documents}\n\nAnd the summary:\n{summary}\n\nEvaluate whether the summary captures both common and unique ideas of the documents."
        evaluation_prompts.append(prompt)

    evaluations = [evaluator(prompt, max_length=500, do_sample=False)[0]['generated_text'] for prompt in evaluation_prompts]
    return evaluations

def main():
    args = parse_args()
    # Sample documents
    documents_df = pd.read_csv(args.summaries)
    reviews_by_doc = documents_df.groupby('id')['text'].apply(list).reset_index()

    for paper in reviews_by_doc['id']:
        # Generate summaries using different methods
        documents = reviews_by_doc[reviews_by_doc['id'] == paper]['text']
        lex_summary = lexrank_summary(documents)
        lsa_summary = lsa_summary(documents)
        bert_summary_text = bert_summary(documents)
        llama_summary_text = llama_summary(documents)

        # Compile all summaries
        summaries = [" ".join(lex_summary), " ".join(lsa_summary), bert_summary_text, llama_summary_text]

        # Evaluate summaries
        evaluations = evaluate_summaries(documents, summaries)

        for i, eval in enumerate(evaluations):
            print(f"Evaluation for Summary {i + 1}:\n{eval}\n")


if __name__ == "__main__":
    main()
