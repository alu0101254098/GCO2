import os
import json
import nltk
import numpy as np
import pandas as pd
import openpyxl as oxl
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('punkt')
# print("punkt download")

# Constantes
COL_INDX = 'Index'
COL_1 = 'Term'
COL_2 = 'TF'
COL_3 = 'IDF'
COL_4 = 'TF_IDF'
RESULT_XLSX = "result.xlsx"


# Función principal
def main(stop_words_file, corpus_file, documents_file):
    try:
        # Cargar las palabras de parada
        stop_words = set()
        with open(stop_words_file, 'r') as stopwords_file:
            stop_words = set(stopwords_file.read().splitlines())
        # print("stop words loaded")

        # Cargar el archivo de lematización
        lemmatization = {}
        with open(corpus_file, 'r') as lemmatization_file:
            lemmatization = json.load(lemmatization_file)
        # print("lematization loaded")
        
        #  Leer el archivo de texto
        with open(documents_file, 'r') as file:
            documents = file.read().splitlines()
        # print("text files readed")

        # Preprocesar los documentos
        preprocessed_documents = [preprocess_text(doc, stop_words, lemmatization) for doc in documents]
        # print("preprocesses_documents done")

        # Calcular TF-IDF
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(preprocessed_documents)
        terms = vectorizer.get_feature_names_out()
        # print("TF-IDF calcualted")

        # Calcular la similaridad del coseno entre documentos
        cosine_similarities = cosine_similarity(tfidf_matrix, tfidf_matrix)
        # print("cosine calculated")

        # DF - Soluciones
        all_results = []

        # Guardar los resultados en un archivo
        # print(len(documents))
        similarity_df = create_similarity_df(len(documents))
        # print(similarity_df)
        for i, doc in enumerate(documents):
            result_df = create_result_df()
            for j, term in enumerate(terms):
                tf = tfidf_matrix[i, j]
                idf = vectorizer.idf_[j]
                tfidf = tf * idf
                values = [j, term, f'{tf:.4f}', f'{idf:.4f}', f'{tfidf:.4f}']
                result_df = add_to_result_df(result_df,values)  
            
            all_results.append(result_df)
            doc_i_label = "Document" + str(i+1)
            pos_label = [doc_i_label, 0]
            for j, similarity in enumerate(cosine_similarities[i]):
                doc_j_label = "Document" + str(j+1)
                pos_label[1] = doc_j_label
                similarity_df = add_similarity(similarity_df, pos_label, float(similarity))
        
        # Devolver la salida 
        output(all_results, similarity_df, documents_file)
        return [all_results, similarity_df]
    
    except Exception as error:
        print(f'ERROR: {error}')
        return -1

# Función para lematizar un término
def lemmatize_term(term, lemmatization):
    try:
        return lemmatization.get(term, term)
    except Exception as error:
        print(f'Exception in lemmatize_term: {error}')
        raise

# Función para preprocesar un texto
def preprocess_text(text, stop_words, lemmatization):
    try:
        tokens = word_tokenize(text)
        tokens = [lemmatize_term(token.lower(), lemmatization) for token in tokens if token.isalpha() and token.lower() not in stop_words]
        return ' '.join(tokens)
    except Exception as error:
        print(f'Exception in preprocess_text: {error}')
        raise

# Crear una tabla con las columnas: Indice, Término, TF, IDF, TF-IDF
def create_result_df():
    try:
        col = [COL_INDX, COL_1, COL_2, COL_3, COL_4]
        result_df = pd.DataFrame(columns=col)
        return result_df    
    except Exception as error:
        print(f'Exception in create_result_df: {error}')
        raise

# Añadir fila a la solucion
def add_to_result_df(result_df, values):
    try:
        temp_df = pd.DataFrame([[values[0], values[1], values[2], values[3], values[4]]]
                               , columns=[COL_INDX, COL_1, COL_2, COL_3, COL_4])

        if result_df.empty:
            result_df = temp_df
        else:
            result_df = pd.concat([result_df, temp_df], ignore_index=True)

        return result_df
    except Exception as error:
        print(f'Exception in add_to_result_df: {error}')
        raise

# Crea el Dataframe para la similitud entre documentos
def create_similarity_df(num_documents):
    try:
        col_row = [f'Document{i+1}' for i in range(num_documents)]
        similarity_df = pd.DataFrame(index = col_row, columns= col_row)

        for i in range(num_documents):
            similarity_df.at[f'Document{i+1}', f'Document{i+1}'] = '#'
        
        return similarity_df

    except Exception as error:
        print(f'Exception in create_similarity_df: {error}')
        raise

# Añade similitudes al dataframe
def add_similarity(similarity_df, pos_label, similarity):
    try:
        if(pos_label[0] == pos_label[1]):
            return similarity_df
        # print(pos_label[0], pos_label[1])
        similarity_df.at[pos_label[0], pos_label[1]] = round(similarity, 4)
        return similarity_df

    except Exception as error:
        print(f'Exception in add_similarity: {error}')
        raise

# Almacena la salida del programa en un fichero xlsx
def final_xlsx_output(all_results, similarity_df, documents_file):
    try:
        document_name = os.path.splitext(documents_file)[0]
        output_file = document_name + '-' + RESULT_XLSX
        with pd.ExcelWriter(output_file, engine='openpyxl', mode='w') as writer:
            similarity_df.to_excel(writer, sheet_name="Similarity DataFrame")
            for i in range(len(all_results)):
                all_results[i].to_excel(writer, sheet_name=f'Document{i+1}', index=False)
        return output_file
    except Exception as error:
        print(f'Exception in final_xlsx_output: {error}')
        raise

# Devuelve la salida total del programa
def output(all_results, similarity_df, documents_file):
    try:
        output_file = final_xlsx_output(all_results, similarity_df, documents_file)
        [print(f"Document {idx + 1}:\n{result}\n") for idx, result in enumerate(all_results)]
        print(similarity_df)
        print(f'\nSe han registrado los resultados completos en el fichero: {output_file}')
    except Exception as error:
        print(f'Exception in output: {error}')
        raise







