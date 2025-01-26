from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words

import argparse

import pandas as pd
from pathlib import Path

import nltk


def summarize(method, language, sentence_count, input_type, input_):
    if method == 'LSA':
        from sumy.summarizers.lsa import LsaSummarizer as Summarizer
    if method == 'text-rank':
        from sumy.summarizers.text_rank import TextRankSummarizer as Summarizer
    if method == 'lex-rank':
        from sumy.summarizers.lex_rank import LexRankSummarizer as Summarizer
    if method == 'edmundson':
        from sumy.summarizers.edmundson import EdmundsonSummarizer as Summarizer
    if method == 'luhn':
        from sumy.summarizers.luhn import LuhnSummarizer as Summarizer
    if method == 'kl-sum':
        from sumy.summarizers.kl import KLSummarizer as Summarizer
    if method == 'random':
        from sumy.summarizers.random import RandomSummarizer as Summarizer
    if method == 'reduction':
        from sumy.summarizers.reduction import ReductionSummarizer as Summarizer

    if input_type == "text":
        parser = PlaintextParser.from_string(input_, Tokenizer(language))

    stemmer = Stemmer(language)
    summarizer = Summarizer(stemmer)
    stop_words = get_stop_words(language)

    if method == 'edmundson':
        summarizer.null_words = stop_words
        summarizer.bonus_words = parser.significant_words
        summarizer.stigma_words = parser.stigma_words
    else:
        summarizer.stop_words = stop_words

    summary_sentences = summarizer(parser.document, sentence_count)
    summary = ' '.join([str(sentence) for sentence in summary_sentences])

    return summary


# methods = ['LSA', 'text-rank', 'lex-rank', 'edmundson', 'luhn', 'kl-sum', 'random', 'reduction']


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="reviews")
    # method
    parser.add_argument("--method", type=str, choices=['LSA', 'text-rank', 'lex-rank', 'edmundson', 'luhn', 'kl-sum', 'random', 'reduction'], default="LSA")
    parser.add_argument("--output", type=Path, default="")

    args = parser.parse_args()
    return args

def prepare_dataset(dataset_name, dataset_path="data/test/"):
    dataset_path = Path(dataset_path)
    if dataset_name == "reviews":
        dataset = pd.read_csv(dataset_path / "samples.csv")
    else:
        raise ValueError(f"Unknown dataset {dataset_name}")


    return dataset


# group text by sample id and concatenate text

def group_text_by_id(df: pd.DataFrame) -> pd.DataFrame:
    """
    Group the text by the sample id and concatenate the text.
    :param df: The dataframe
    :return: The dataframe with the text grouped by the sample id
    """
    texts = df.groupby("id")["text"].apply(lambda x: " ".join(x))

    # retrieve first gold by id
    gold = df.groupby("id")["gold"].first()

    # create new dataframe
    df = pd.DataFrame({"text": texts, "gold": gold}, index=texts.index)

    return df


def main():
    args = parse_args()
    for N in [1]:
        dataset = prepare_dataset(args.dataset)
        # dataset = group_text_by_id(dataset)

        summaries = []
        for text in dataset.text:
            summary = summarize(args.method, "english", N, "text", text)
            summaries.append(summary)

        dataset['summary'] = summaries
        dataset['metadata/dataset'] = args.dataset
        dataset["metadata/method"] = args.method
        dataset["metadata/sentence_count"] = N

        name = f"{args.dataset}-_-{args.method}-_-sumy_{N}.csv"
        path = Path(args.output) / name

        Path(args.output).mkdir(exist_ok=True, parents=True)
        dataset.to_csv(path, index=True)


main()