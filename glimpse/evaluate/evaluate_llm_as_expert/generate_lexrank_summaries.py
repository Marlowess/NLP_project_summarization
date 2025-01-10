from lexrank import LexRank
from lexrank.mappings.stopwords import STOPWORDS
from nltk.tokenize import sent_tokenize
from path import Path
import nltk

def lexrank_summary(documents):
    """Summarize using LexRank."""

    corpus = []
    corpus_dir = Path('/content/bbc/tech')

    for file_path in corpus_dir.files('*.txt'):
        with file_path.open(mode='rt', encoding='utf-8') as fp:
            corpus.append(fp.readlines())

    sentences = []
    for doc in documents:
        # Tokenize into sentences
        doc_sentences = sent_tokenize(doc)
        # Filter out non-informative sentences
        filtered_sentences = [s for s in doc_sentences if len(s.split()) > 5]
        sentences.extend(filtered_sentences)
    
    # Check if there are enough sentences
    if not sentences:
        return "Unable to generate summary due to lack of informative content."
    
    # LexRank
    try:
        lexrank = LexRank(corpus, stopwords=STOPWORDS['en'])
        summary = lexrank.get_summary(sentences, summary_size=3)
        return " ".join(summary)
    except ValueError as e:
        return f"Error generating summary: {e}"

def main():
    nltk.download('punkt')
    nltk.download('punkt_tab')
    lexrank_summary()
    # todo save summaries
    
if __name__ == "__main__":
    main()