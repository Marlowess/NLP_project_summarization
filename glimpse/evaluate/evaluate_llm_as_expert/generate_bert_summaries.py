from summarizer import Summarizer

def bert_summary(documents):
    """Summarize using BERT-based extractive summarization."""
    model = Summarizer()
    summaries = [model(doc, num_sentences=3) for doc in documents]
    return " ".join(summaries)

def main():
    bert_summary()
    # todo save summaries
    
if __name__ == "__main__":
    main()