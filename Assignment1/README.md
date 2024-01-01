# Inverted Index, Boolean Document Retrieval

Program to create an inverted index and use it to retrieve document IDs as per the supplied query

This project is part of an Information Retrieval (CS60092) assignment at IIT Kharagpur. (A PDF of the work can be found in the repository)

#### Programming Languages Used
* Python (Version - 3.8.10)

#### Libraries Used
* re
* sys
* string
* nltk
* pickle
* NumPy
In case of any missing library, kindly install it using 
    - pip3 install < library name > (for Python)
(Some libraries mentioned above come as part of python3)

### Role of Assignment1_23CS60R22_indexer.py
It takes the 'cran.all.1400' file as input and creates an inverted index using .I (document id) and .W (document text) part of each document. Stop words and punctuation marks are removed from each document text. Then, lemmatization is performed on the document text. This processed document text is used to create an inverted index. The inverted index is then saved as 'model_queries_23CS60R22.bin' using the pickle library.

### Role of Assignment1_23CS60R22_parser.py
It takes 'cran.qry' file as input and extracts .I (query id) and .W (query text) part of each query. Stop words and punctuation marks are removed from each query text. Then, lemmatization is performed on the query text. The query ID and processed query text are saved in the file 'queries_23CS60R22.txt'.

### Role of Assignment1_23CS60R22_bool.py
It takes the 'model_queries_23CS60R22.bin' and 'queries_23CS60R22.txt' files as input. For each query, it fetches the relevant document using the inverted index. The result is then saved in file 'Assignment1_23CS60R22_results.txt.'

## Running it locally on your machine

1. Clone this repository and cd to the project root.
2. Run Assignment1_23CS60R22_indexer.py using - python Assignment1_23CS60R22_indexer.py cran.all.1400
3. Run Assignment1_23CS60R22_parser.py using - python Assignment1_23CS60R22_parser.py cran.qry
4. Run Assignment1_23CS60R22_bool.py using - python Assignment1_23CS60R22_bool.py model_queries_23CS60R22.bin queries_23CS60R22.txt