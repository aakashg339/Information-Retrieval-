# Scoring and Evaluation

Program to score and evaluate the results of queries in Assignment 1.

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

### Role of Assignment2_23CS60R22_ranker.py
It takes 'cran.all.1400' (documents), 'cran.qry' (queries) and 'model_queries_23CS60R22.bin' (inverted index) files as input and finds cosine similarity between queries and documents using three schemes and writes the result in the output file.
* lnc.ltc (Output file - Assignment2_23CS60R22_ranked_list_A.txt)
* lnc.Ltc (Output file - Assignment2_23CS60R22_ranked_list_B.txt)
* anc.apc (Output file - Assignment2_23CS60R22_ranked_list_C.txt)

Refer to the table in the image below for details
![TF-IDF weighting variants](https://slideplayer.com/slide/238597/1/images/41/tf-idf+weighting+has+many+variants.jpg)


### Role of Assignment2_23CS60R22_evaluator.py
It takes 'cranqrel' (gold standard relevance file) and compares against 'Assignment2_23CS60R22_ranked_list_< A, B or C >.txt'. 
Evaluation metrics mentioned below were calculated and written to file 'Assignment2_23CS60R22_metrics_< A, B or C >.txt'
* Average Precision (AP) @ 10
* Average Precision (AP) @ 20
* Normalized Discounted Cumulative Gain (NDCG) @ 10
* Normalized Discounted Cumulative Gain (NDCG) @ 20
* Mean Average Precision (MAP) @ 10
* Mean Average Precision (MAP) @ 20
* Mean Normalized Discounted Cumulative Gain (NDCG) @ 10
* Mean Normalized Discounted Cumulative Gain (NDCG) @ 20

## Running it locally on your machine

1. Clone this repository and cd to the project root.
2. Run Assignment2_23CS60R22_ranker.py using - python3 Assignment2_23CS60R22_ranker.py cran.all.1400 cran.qry model_queries_23CS60R22.bin
3. Run Assignment2_23CS60R22_evaluator.py using - python3 Assignment2_23CS60R22_evaluator.py cranqrel Assignment2_23CS60R22_ranked_list_< A, B, or C >.txt