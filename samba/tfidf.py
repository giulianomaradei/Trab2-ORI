import sys
import spacy
import math
from collections import defaultdict

# Load the spacy model for stopword removal and lemmatization
nlp = spacy.load("pt_core_news_sm")

def preprocess_text(text):
    # Use spacy to remove stopwords and lemmatize
    doc = nlp(text)
    return [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]

def calculate_tf(doc_terms):
    tf = defaultdict(float)
    for term in doc_terms:
        tf[term] += 1.0
    return tf

def calculate_idf(documents):
    idf = defaultdict(float)
    total_docs = len(documents)
    for doc_terms in documents:
        for term in set(doc_terms):
            idf[term] += 1.0
    for term in idf:
        try:
            idf[term] = math.log10(total_docs / idf[term])
        except (ValueError, TypeError, ZeroDivisionError):
            idf[term] = 0.0
    return idf

def calculate_tfidf(tf, idf):
    tfidf = {}
    for term, freq in tf.items():
        try:
            if freq > 0 and term in idf:
                tfidf[term] = (1 + math.log10(freq)) * idf[term]
        except (ValueError, TypeError, ZeroDivisionError):
            continue
    return tfidf

def generate_index_file(documents, document_paths):
    inverted_index = defaultdict(lambda: defaultdict(int))

    for doc_id, doc_terms in enumerate(documents):
        for term in doc_terms:
            inverted_index[term][doc_id + 1] += 1  # doc_id + 1 to match the 1-based index

    with open('indice.txt', 'w') as index_file:
        for term, doc_dict in sorted(inverted_index.items()):
            doc_entries = " ".join(f"{doc_id},{count}" for doc_id, count in sorted(doc_dict.items()))
            index_file.write(f"{term}: {doc_entries}\n")

def main(base_file):
    with open(base_file, 'r') as f:
        document_paths = f.read().splitlines()

    documents = []
    for path in document_paths:
        with open(path, 'r') as doc_file:
            text = doc_file.read()
            terms = preprocess_text(text)
            documents.append(terms)

    idf = calculate_idf(documents)

    # Generate the index file
    generate_index_file(documents, document_paths)

    with open('pesos.txt', 'w') as pesos_file:
        for i, doc_terms in enumerate(documents):
            tf = calculate_tf(doc_terms)
            tfidf = calculate_tfidf(tf, idf)
            non_zero_tfidf = {term: weight for term, weight in tfidf.items()
                            if isinstance(weight, (int, float)) and weight > 0}
            if non_zero_tfidf:
                file_name = document_paths[i]
                pesos_file.write(f"{file_name}: " + " ".join(f"{term},{weight:.6f}"
                    for term, weight in non_zero_tfidf.items()) + "\n")

if __name__ == "__main__":
    main(sys.argv[1])
