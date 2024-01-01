# Information Retrieval Assignment

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

## Assignment 1
We created an inverted index from the documents. Using the query, fetched the relevant documents from the inverted index.

## Assignment 2
The results of the query are evaluated against gold standard relevance.

## Other Details
* Length of the inverted index (Vocabulary) - 8813
* Number of queries that returned document IDs - 14 out of 225

Few evaluation metrics
* lnc.ltc
    1. Mean Average Precision (MAP) @ 10 = 0.4509512345679012
    2. Mean Average Precision (MAP) @ 20 = 0.40917921521906336
    3. Mean Normalized Discounted Cumulative Gain (NDCG) @ 10 = 0.35613102024509247
    4. Mean Normalized Discounted Cumulative Gain (NDCG) @ 20 = 0.387901807896173
* lnc.Ltc
    1. Mean Average Precision (MAP) @ 10 = 0.4509512345679012
    2. Mean Average Precision (MAP) @ 20 = 0.40917921521906336
    3. Mean Normalized Discounted Cumulative Gain (NDCG) @ 10 = 0.35613102024509247
    4. Mean Normalized Discounted Cumulative Gain (NDCG) @ 20 = 0.387901807896173
* anc.apc
    1. Mean Average Precision (MAP) @ 10 = 0.4321748383303936
    2. Mean Average Precision (MAP) @ 20 = 0.3955611461367082
    3. Mean Normalized Discounted Cumulative Gain (NDCG) @ 10 = 0.34492256327874254
    4. Mean Normalized Discounted Cumulative Gain (NDCG) @ 20 = 0.37787975030561444

### Purpose
To understand how to create an inverted index, retrieve relevant documents as part of the query and various evaluation metrics.

### References
1. http://web.stanford.edu/class/cs276/
2. https://slideplayer.com/slide/238597/1/images/41/tf-idf+weighting+has+many+variants.jpg