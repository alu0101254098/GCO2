import os
import json
import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('punkt')
print("punkt download")

COL_0 = 'Term'
COL_1 = 'TF'
COL_2 = 'IDF'
COL_3 = 'TF_IDF'

# Función para lematizar un término
def lemmatize_term(term):
    return lemmatization.get(term, term)

# Función para preprocesar un texto
def preprocess_text(text):
    tokens = word_tokenize(text)
    tokens = [lemmatize_term(token.lower()) for token in tokens if token.isalpha() and token.lower() not in stop_words]
    return ' '.join(tokens)

# Transformar la salida en tablas: 
def create_result_df():
    try:
        col = [COL_0, COL_1, COL_2,COLÇ]
        result_df = pd.Dataframe(columns=col)
        return result_df    
    except Exception as error:
        print(f'Exception in create_result_df: {error}')
        raise

# Añadir fila a la solucion
def add_to_result_df(values):
    try:
        temp_df = pd.DataFrame([[values[0], values[1], values[2], values[3]]]
                               , columns=[COL_0, COL_1, COL_2, COL_3])

        if result_df.empty:
            result_df = temp_df
        else:
            result_df = pd.concat([result_df, temp_df], ignore_index=True)

        return result_df
    except Exception as error:
        print(f'Exception in add_to_result_df: {error}')
        raise

def create_similarity_df(num_documents):
    try:
        col_row = [f'Document{i}' for i in range(num_documents)]
        similarity_df = pd.DataFrame(index = col_row, columns= col_row)

        for i in range(num_documents):
            similarity_df.at[f'Document{i}', f'Document{i}'] = '#'

    except Exception as error:
        print(f'Exception in create_similarity_df: {error}')
        raise

# Cargar las palabras de parada
stop_words = set()
with open('stop-words-en.txt', 'r') as stopwords_file:
    stop_words = set(stopwords_file.read().splitlines())
print("stop words loaded")

# Cargar el archivo de lematización
lemmatization = {}
with open('corpus-en.txt', 'r') as lemmatization_file:
    lemmatization = json.load(lemmatization_file)
print("lematization loaded")
# Leer el archivo de texto
with open('documents-01.txt', 'r') as file:
    documents = file.read().splitlines()
print("text files readed")

# Preprocesar los documentos
preprocessed_documents = [preprocess_text(doc) for doc in documents]
print("preprocesses_documents done")

# Calcular TF-IDF
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(preprocessed_documents)
terms = vectorizer.get_feature_names_out()

print("TF-IDF calcualted")

# Calcular la similaridad del coseno entre documentos
cosine_similarities = cosine_similarity(tfidf_matrix, tfidf_matrix)
print("cosine calculated")

# Guardar los resultados en un archivo
with open('output.txt', 'w') as output_file:
    list_results_df = {}
    for i, doc in enumerate(documents):
        output_file.write(f'Document {i + 1}:\n')
        output_file.write('Index | Term | TF | IDF | TF-IDF\n')
        for j, term in enumerate(terms):
            tf = tfidf_matrix[i, j]
            idf = vectorizer.idf_[j]
            tfidf = tf * idf
            output_file.write(f'{j} | {term} | {tf:.4f} | {idf:.4f} | {tfidf:.4f}\n')
            values = [term, f'{tf:.4f}', f'{idf:.4f}', f'{tfidf:.4f}']

        
        
        

        output_file.write('Similitud de coseno entre documentos:\n')
        for j, similarity in enumerate(cosine_similarities[i]):
            output_file.write(f'Document {j + 1}: {similarity:.4f}\n')
        output_file.write('-' * 50 + '\n')



result_df = create_result_df()



