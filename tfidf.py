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
    tf = defaultdict(int)
    for term in doc_terms:
        tf[term] += 1
    return tf

def calculate_idf(documents):
    idf = defaultdict(int)
    total_docs = len(documents)
    for doc_terms in documents:
        for term in set(doc_terms):
            idf[term] += 1
    for term in idf:
        idf[term] = math.log10(total_docs / idf[term])
    return idf

def calculate_tfidf(tf, idf):
    tfidf = {}
    for term, freq in tf.items():
        if freq > 0:
            tfidf[term] = (1 + math.log10(freq)) * idf[term]
    return tfidf

def generate_index_file(documents, document_paths):
    with open('indice.txt', 'w') as index_file:
        for i, doc_terms in enumerate(documents):
            unique_terms = set(doc_terms)
            index_file.write(f"doc{i+1}.txt: " + " ".join(unique_terms) + "\n")

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
            non_zero_tfidf = {term: weight for term, weight in tfidf.items() if weight > 0}
            if non_zero_tfidf:
                pesos_file.write(f"doc{i+1}.txt: " + " ".join(f"{term}, {weight:.4f}" for term, weight in non_zero_tfidf.items()) + "\n")

if __name__ == "__main__":
    main(sys.argv[1])
