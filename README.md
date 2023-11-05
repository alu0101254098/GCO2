# **Sistema de recomendación**
## **Modelos basados en el contenido**
> Participantes:
   - Eduardo González Pérez
   - Jonathan Martínez Pérez

## Contenido
  - [Instrucciones de instalación de dependencias](#id0)
  - [Descripción del código desarrollado](#id1)
  - [Ejemplo de uso](#id2)

## Instrucciones de instalación de dependencias<a name="id0"></a>
Para poder emplear nuestro código, se tendrá que realizar un `git clone` de este repositorio y contar, en la máquina donde ejecutaremos el mismo, los siguientes elementos:
- __Python__: nuestro código se ha realizado en el lenguaje de programación Python. Por este motivo, será necesario tener instalado ___Python3___. En caso de no tenerlo, si nos encontramos en una máquina Linux, se debería realizar los siguientes comandos: `sudo apt update`  y  `sudo apt install python3`.
  
- __Librerías__: para la realización de esta práctica se emplea las siguientes librerías, las cuáles podremos instalar mediante el comando `pip install [libreria]`:
  - ___numpy___ : soporte para matrices multidimensionales y colección de funciones matemáticas de alto nivel.
  - ___pandas___ : especializada en la manipulación y el análisis de datos.
  - ___argparse___ : procesamiento de la línea de comando de forma flexible y con funcionalidades adicionales.
  - ___openpyxl___ : lectura y escritura de ficheros excel xlsx/xlsm/xltx/xltm.
  - ___nltk___ : diseñada para el procesamiento del lenguaje natural (NLP).
  - ___sklearn___ : nos ayuda con los cálculos del IDF, TF y la similarifad  de coseno.
  - ___json___ : funciones para trabajar con datos en formato JSON (JavaScript Object Notation), especificamente para procesar el corpus.

## Descripción del código desarrollado<a name="id1"></a>

### Dentro de `functions_recommender_bc.py`:
Tenemos un código desarrolldo en lenguaje Python, este recibe un fichero `stopwords.txt`, el cual nos indica las palabras que deben ignorarse, `corpus.txt`, que nos ayuda a pasar los verbos en sus diferentes tiempos a infinitivo.
Finalmente, los ficheros de texto plano que les pasemos estaran compuestos por diferentes líneas las cuales cada una representa un documento, es decir 10 líneas serían 10 documentos.
`main(stop_words_file, corpus_file, documents_file)`: Esta es la función principal que coordina todo el proceso. Recibe tres archivos como entrada: un archivo de palabras de parada, un archivo de lematización y un archivo de documentos de texto. Luego, realiza los siguientes pasos:

- Carga las palabras de parada desde el archivo.
- Carga la lematización desde el archivo.
- Lee el archivo de documentos de texto.
- Preprocesa los documentos de texto para eliminar palabras de parada y lematizar términos.
- Calcula la matriz TF-IDF para los documentos.
- Calcula la similitud de coseno entre los documentos.
- Crea un archivo de salida en formato XLSX que contiene los resultados.

`lemmatize_term(term, lemmatization)`: Esta función toma un término y un diccionario de lematización como entrada y devuelve la forma lematizada del término si está presente en el diccionario. Se utiliza para normalizar los términos en el proceso de preprocesamiento.

`preprocess_text(text, stop_words, lemmatization)`: Esta función toma un texto, un conjunto de palabras de parada y un diccionario de lematización como entrada. Realiza el preprocesamiento del texto, que incluye la tokenización, la eliminación de palabras de parada y la lematización de los términos. Luego, devuelve el texto preprocesado.

`create_result_df()`: Esta función crea un DataFrame vacío con las columnas: 'Indice', 'Término', 'TF', 'IDF', 'TF-IDF'. El DataFrame se utiliza para almacenar los resultados de TF, IDF y TF-IDF para los términos.

`add_to_result_df(result_df, values)`: Esta función agrega una fila al DataFrame de resultados con los valores proporcionados en la lista values. Si el DataFrame está vacío, crea uno nuevo y lo devuelve con la fila agregada.

`create_similarity_df(num_documents)`: Esta función crea un DataFrame vacío para almacenar la similitud de coseno entre los documentos. El número de documentos se especifica como entrada y se utiliza para crear las filas y columnas correspondientes en el DataFrame.

`add_similarity(similarity_df, pos_label, similarity)`: Esta función agrega un valor de similitud de coseno al DataFrame de similitud de coseno. pos_label contiene las etiquetas de los documentos para los que se calcula la similitud y similarity es el valor de similitud. No agrega valores si los documentos son iguales para evitar redundancia.

`final_xlsx_output(all_results, similarity_df)`: Esta función almacena la salida del programa en un archivo XLSX. Guarda el DataFrame de similitud de coseno y los DataFrames de resultados para cada documento en hojas separadas del archivo XLSX.

`output(all_results, similarity_df)`: Esta función es responsable de imprimir los resultados en la consola y luego llamar a final_xlsx_output para guardar los resultados en un archivo XLSX.

### Dentro de `main_recommender_bc.py`

En este fichero tenemos el módulo argparse para parsear argumentos de línea de comandos y luego llama a la función main del módulo functions_recommender_bc con los argumentos proporcionados.

## Ejemplo de uso<a name="id2"></a>

python3 script.py -s my_stop_words.txt -c my_corpus.txt -d my_documents.txt
