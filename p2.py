import os
import json
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('punkt')

# Función para lematizar un término
def lemmatize_term(term):
    return lemmatization.get(term, term)

# Función para preprocesar un texto
def preprocess_text(text):
    tokens = word_tokenize(text)
    tokens = [lemmatize_term(token.lower()) for token in tokens if token.isalpha() and token.lower() not in stop_words]
    return ' '.join(tokens)

# Cargar las palabras de parada
stop_words = set()
with open('stopwords.txt', 'r') as stopwords_file:
    stop_words = set(stopwords_file.read().splitlines())

# Cargar el archivo de lematización
lemmatization = {}
with open('corpus.txt', 'r') as lemmatization_file:
    lemmatization = json.load(lemmatization_file)

# Leer el archivo de texto
with open('input.txt', 'r') as file:
    documents = file.read().splitlines()

# Preprocesar los documentos
preprocessed_documents = [preprocess_text(doc) for doc in documents]

# Calcular TF-IDF
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(preprocessed_documents)
terms = vectorizer.get_feature_names_out()

# Calcular la similaridad del coseno entre documentos
cosine_similarities = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Guardar los resultados en un archivo
with open('output.txt', 'w') as output_file:
    for i, doc in enumerate(documents):
        output_file.write(f'Document {i + 1}:\n')
        output_file.write('Index | Term | TF | IDF | TF-IDF\n')
        for j, term in enumerate(terms):
            tf = tfidf_matrix[i, j]
            idf = vectorizer.idf_[j]
            tfidf = tf * idf
            output_file.write(f'{j} | {term} | {tf:.4f} | {idf:.4f} | {tfidf:.4f}\n')
        
        output_file.write('Coseno Similarity with other documents:\n')
        for j, similarity in enumerate(cosine_similarities[i]):
            output_file.write(f'Document {j + 1}: {similarity:.4f}\n')
        output_file.write('-' * 50 + '\n')
