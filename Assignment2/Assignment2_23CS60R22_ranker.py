import re
import sys
import string
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pickle
import numpy as np

# Class to build inverted index from the given data
class TF_IDFVectorizationAndEvaluation:

    # Defining constants
    FILE_RANKED_LIST_lnc_ltd = "Assignment2_23CS60R22_ranked_list_A.txt"
    FILE_RANKED_LIST_lnc_Ltc = "Assignment2_23CS60R22_ranked_list_B.txt"
    FILE_RANKED_LIST_anc_apc = "Assignment2_23CS60R22_ranked_list_C.txt"

    def __init__(self, inputFileInvertedIndex, inputFileDocument, inputFileQuery):
        # File names
        self.inputFileInvertedIndex = inputFileInvertedIndex
        self.inputFileDocument = inputFileDocument
        self.inputFileQuery = inputFileQuery

        # Data structures
        self.invertedIndex = {}
        self.df = {}
        self.iDf = {}
        self.processedDocuments = []
        self.iDDocumentMap = {}
        self.preProcessedDocuments = []
        self.preProcessedDocumentsIds = []
        self.preProcessedQueries = []
        self.preProcessedQueriesIds = []
        self.defaultLemmatizer = WordNetLemmatizer()
        self.defaultStopWords = set(stopwords.words('english'))

        # TF-IDF vector related data structures
        self.tFVectorDocument = []
        self.tFVectorQuery = []
        self.tF_IDFVectorDocument = []
        self.tF_IDFVectorQuery = []
        self.tF_IDFVectorDocument_lncltc = []
        self.tF_IDFVectorQuery_lncltc = []
        self.tF_IDFVectorDocument_lncLtc = []
        self.tF_IDFVectorQuery_lncLtc = []
        self.tF_IDFVectorDocument_ancapc = []
        self.tF_IDFVectorQuery_ancapc = []

    # Function to get the inverted index from file
    def loadInvertedIndex(self):
        try:
            with open(self.inputFileInvertedIndex, 'rb') as invertedIndexFile:
                self.invertedIndex = pickle.load(invertedIndexFile)
        except FileNotFoundError:
            sys.exit("File " + self.inputFileInvertedIndex + " not found. Kindly check the file name")
    
    # Function to get the document frequency for each word in inverted index
    def getDocumentFrequency(self):
        for token, postingsList in self.invertedIndex.items():
            self.df[token] = len(postingsList)

    # Function to read data from input file
    def createProcessedDocuments(self, documentOrQuery):
        # Reading from the given input file
        if documentOrQuery == 'document':
            inputFile = self.inputFileDocument
        elif documentOrQuery == 'query':
            inputFile = self.inputFileQuery

        try:
            with open(inputFile, 'r') as file:
                inputFileData = file.read()
        except FileNotFoundError:
            sys.exit("File " + self.inputFile + " not found. Kindly check the file name")

        # Spliting the data based on .I
        documents = re.split(r'(\.I \d+)', inputFileData)

        # Remove extra spaces
        self.processedDocuments = []
        for document in documents:
            strippedDocument = document.strip()
            if strippedDocument:
                self.processedDocuments.append(strippedDocument)
    
    # Fuction to create the id and text data map
    def createIdDocumentMap(self):
        documentId = None
        documentText = None
        self.iDDocumentMap = {}
        for processedDocument in self.processedDocuments:
            if '.I' in processedDocument:
                documentId = int(re.search(r'\.I (\d+)', processedDocument).group(1))
            elif '.W' in processedDocument:
                documentText = processedDocument[processedDocument.find('.W')+2:]  # find('.W')+2 is included to not include .W in result
                documentText = documentText.replace('\n', ' ').lower()
                if documentId is not None and documentText is not None:
                    self.iDDocumentMap[documentId] = documentText
                    documentId = None
                    documentText = None
            else :
                print("Either Id or document not matching")

    # Function to remove the punctuation marks and convert the words to lower case
    def removePunctuationConvertLower(self, wordTokens):
        wordTokensPunctuationRemoved = []
        for wordToken in wordTokens:
            if wordToken not in string.punctuation:
                wordTokensPunctuationRemoved.append(wordToken.lower())
        return wordTokensPunctuationRemoved
    
    # Function to remove stop words if any
    def removeStopWords(self, wordTokensPunctuationRemoved):
        wordTokensStopWordsRemoved = []
        for wordToken in wordTokensPunctuationRemoved:
                if wordToken not in self.defaultStopWords:
                    wordTokensStopWordsRemoved.append(wordToken)
        return wordTokensStopWordsRemoved
    
    # Function to lemmatize words
    def lemmatizeWords(self, wordTokensStopWordsRemoved):
        lemmatizedWords = []
        for wordToken in wordTokensStopWordsRemoved:
            lemmatizedWords.append(self.defaultLemmatizer.lemmatize(wordToken))
        return lemmatizedWords

    # Helper method to get lemmatized tokens
    def processDocument(self, documentText):
        wordTokens = word_tokenize(documentText)

        wordTokensPunctuationRemoved = self.removePunctuationConvertLower(wordTokens)
        
        wordTokensStopWordsRemoved = self.removeStopWords(wordTokensPunctuationRemoved)
        
        lemmatizedTokens = self.lemmatizeWords(wordTokensStopWordsRemoved)

        return lemmatizedTokens

    # Function to create map of pre-processed documents
    def createPreProcessedDocuments(self):
        for documentId, documentText in self.iDDocumentMap.items():
            # Get lemmatized words for each text data
            processedDocumentText = self.processDocument(documentText)

            # Create preprocessed documents
            self.preProcessedDocuments.append(processedDocumentText)
            self.preProcessedDocumentsIds.append(documentId)

    # Function to create map of pre-processed queries
    def createPreProcessedQueries(self):
        for documentId, documentText in self.iDDocumentMap.items():
            # Get lemmatized words for each text data
            processedQueryText = self.processDocument(documentText)

            # Create preprocessed documents
            self.preProcessedQueries.append(processedQueryText)
            self.preProcessedQueriesIds.append(documentId)
    
    # Function to create inverse document frequency for each word in the inverted index
    def createInverseDocumentFrequency(self):
        numberOfDocuments = len(self.preProcessedDocuments)
        for word, documentFrequency in self.df.items():
            self.iDf[word] = np.log(numberOfDocuments/documentFrequency)
    
    # Function to create TF vector for documents
    def createTFVectorDocument(self):
        # Create TF vector for each document
        self.tFVectorDocument = {}
        for index, string in enumerate(self.preProcessedDocuments):
            
            # Calculating frequency for each word in the document
            tfForEachtext = {}
            for word in string:
                if word not in tfForEachtext:
                    tfForEachtext[word] = 1
                else:
                    tfForEachtext[word] += 1
            
            self.tFVectorDocument[index+1] = tfForEachtext
            
        #print(self.tFVectorDocument)

    # Function to create TF vector for queries
    def createTFVectorQuery(self):
        # Create TF vector for each query
        self.tFVectorQuery = {}
        for index, string in enumerate(self.preProcessedQueries):

            queryId = self.preProcessedQueriesIds[index]
            
            # Calculating frequency for each word in the document
            # Adding those words which are present in Inverted Index
            tfForEachtext = {}
            for word in string:
                if word in self.df:
                    if word not in tfForEachtext:
                        tfForEachtext[word] = 1
                    else:
                        tfForEachtext[word] += 1
            
            self.tFVectorQuery[queryId] = tfForEachtext
            
        #print(self.tFVectorQuery)

    # Function to create TF-IDF vector for documents and queries in lnc.ltc format
    def createTF_IDFVectorDocument_lncltc(self):
        self.tF_IDFVectorDocument_lncltc = self.tFVectorDocument.copy()
        self.tF_IDFVectorQuery_lncltc = self.tFVectorQuery.copy()

        # Create TF-IDF vector for each document
        for document in self.tFVectorDocument:
            self.tF_IDFVectorDocument_lncltc[document] = self.tFVectorDocument[document].copy()
            cosine_normalization_document = 0.0

            # Processing term frequency as per l scheme
            for token in self.tF_IDFVectorDocument_lncltc[document]:
                self.tF_IDFVectorDocument_lncltc[document][token] = 1 + np.log10(self.tF_IDFVectorDocument_lncltc[document][token])
                cosine_normalization_document += self.tF_IDFVectorDocument_lncltc[document][token] ** 2

            # Cosine Normalization
            for token in self.tF_IDFVectorDocument_lncltc[document]:
                self.tF_IDFVectorDocument_lncltc[document][token] /= np.sqrt(cosine_normalization_document)
        
        # Create TF-IDF vector for each query
        for query in self.tFVectorQuery:
            self.tF_IDFVectorQuery_lncltc[query] = self.tFVectorQuery[query].copy()
            cosine_normalization_query = 0.0

            # Processing term frequency as per l scheme
            for token in self.tF_IDFVectorQuery_lncltc[query]:
                self.tF_IDFVectorQuery_lncltc[query][token] = 1 + np.log10(self.tF_IDFVectorQuery_lncltc[query][token])

                # TF * IDF
                self.tF_IDFVectorQuery_lncltc[query][token] *= self.iDf[token]
                cosine_normalization_query += self.tF_IDFVectorQuery_lncltc[query][token] ** 2

            # Cosine Normalization
            for token in self.tF_IDFVectorQuery_lncltc[query]:
                self.tF_IDFVectorQuery_lncltc[query][token] /= np.sqrt(cosine_normalization_query)
    
    # Function to create TF-IDF vector for documents and queries in lnc.Ltc format
    def createTF_IDFVectorDocument_lncLtc(self):
        self.tF_IDFVectorDocument_lncLtc = self.tFVectorDocument.copy()
        self.tF_IDFVectorQuery_lncLtc = self.tFVectorQuery.copy()

        # ---- Computations for document vectors according to lnc scheme
        for document in self.tFVectorDocument:
            self.tF_IDFVectorDocument_lncLtc[document] = self.tFVectorDocument[document].copy()
            cosine_normalization_document = 0.0

            # Processing term frequency as per l scheme
            for token in self.tF_IDFVectorDocument_lncLtc[document]:
                self.tF_IDFVectorDocument_lncLtc[document][token] = 1 + np.log10(self.tF_IDFVectorDocument_lncLtc[document][token])
                cosine_normalization_document += self.tF_IDFVectorDocument_lncLtc[document][token] ** 2

            # Cosine Normalization
            for token in self.tF_IDFVectorDocument_lncLtc[document]:
                self.tF_IDFVectorDocument_lncLtc[document][token] /= np.sqrt(cosine_normalization_document)

        # ---- Computations for query vectors according to Ltc scheme
        for query in self.tF_IDFVectorQuery_lncLtc:
            self.tF_IDFVectorQuery_lncLtc[query] = self.tFVectorQuery[query].copy()
            cosine_normalization_query = 0.0

            # Processing term frequency as per L scheme
            average_term_frequency = 0.0
            for token in self.tF_IDFVectorQuery_lncLtc[query]:
                average_term_frequency += self.tF_IDFVectorQuery_lncLtc[query][token]
            average_term_frequency /= len(self.tF_IDFVectorQuery_lncLtc[query])

            for token in self.tF_IDFVectorQuery_lncLtc[query]:
                self.tF_IDFVectorQuery_lncLtc[query][token] = (1 + np.log10(self.tF_IDFVectorQuery_lncLtc[query][token])) / (1 + np.log10(average_term_frequency))

                # TF * IDF
                self.tF_IDFVectorQuery_lncLtc[query][token] *= self.iDf[token]
                cosine_normalization_query += self.tF_IDFVectorQuery_lncLtc[query][token] ** 2

            # Cosine Normalization
            for token in self.tF_IDFVectorQuery_lncLtc[query]:
                self.tF_IDFVectorQuery_lncLtc[query][token] /= np.sqrt(cosine_normalization_query)

    # Function to create TF-IDF vector for documents and queries in ancapc format
    def createTF_IDFVectorDocument_ancapc(self):
        self.tF_IDFVectorDocument_ancapc = self.tFVectorDocument.copy()
        self.tF_IDFVectorQuery_ancapc = self.tFVectorQuery.copy()
        n = len(self.tFVectorDocument)
        pIDF = self.df.copy()

        # ---- Computations for document vectors according to anc scheme
        for document in self.tFVectorDocument:
            self.tF_IDFVectorDocument_ancapc[document] = self.tFVectorDocument[document].copy()
            cosine_normalization_document = 0.0

            # Processing term frequency as per a scheme
            maximum_document_term_frequency = 0.0
            if len(self.tFVectorDocument[document].values()) > 0:
                maximum_document_term_frequency = max(self.tFVectorDocument[document].values())

            for token in self.tF_IDFVectorDocument_ancapc[document]:
                self.tF_IDFVectorDocument_ancapc[document][token] = 0.5 + (
                            0.5 * self.tF_IDFVectorDocument_ancapc[document][token]) / maximum_document_term_frequency
                cosine_normalization_document += self.tF_IDFVectorDocument_ancapc[document][token] ** 2

            # Cosine Normalization
            for token in self.tF_IDFVectorDocument_ancapc[document]:
                self.tF_IDFVectorDocument_ancapc[document][token] /= np.sqrt(cosine_normalization_document)

        # ---- Computations for query vectors according to apc scheme

        # Processing document frequency as per p scheme
        for token in pIDF:
            pIDF[token] = max(0.0, np.log10((n / pIDF[token]) - 1))

        for query in self.tFVectorQuery:
            self.tF_IDFVectorQuery_ancapc[query] = self.tFVectorQuery[query].copy()
            cosine_normalization_query = 0.0

            # Processing term frequency as per a scheme
            maximum_query_term_frequency = 0.0
            if len(self.tFVectorQuery[query].values()) > 0:
                maximum_query_term_frequency = max(self.tFVectorQuery[query].values())

            for token in self.tF_IDFVectorQuery_ancapc[query]:
                self.tF_IDFVectorQuery_ancapc[query][token] = 0.5 + (
                            0.5 * self.tF_IDFVectorQuery_ancapc[query][token]) / maximum_query_term_frequency

                # TF * IDF
                self.tF_IDFVectorQuery_ancapc[query][token] *= pIDF[token]
                cosine_normalization_query += self.tF_IDFVectorQuery_ancapc[query][token] ** 2

            # Cosine Normalization
            for token in self.tF_IDFVectorQuery_ancapc[query]:
                self.tF_IDFVectorQuery_ancapc[query][token] /= np.sqrt(cosine_normalization_query)

    # Function to rank documents based on cosine similarity.
    def rankDocuments(self, tF_IDFVectorDocument, tF_IDFVectorQuery, rankedDocumentsOutputFileName):
        # Calculate cosine similarity for each document and each query and store in csv file

        outputData = ""

        # Prepare data to be written to file
        for query in tF_IDFVectorQuery:
            document_vector = []
            for document in tF_IDFVectorDocument:
                dot_product = 0.0

                # Calculate ranking by dot product of each tokens in query and document
                for token in tF_IDFVectorQuery[query]:
                    if token in tF_IDFVectorDocument[document]:
                        dot_product += tF_IDFVectorQuery[query][token] * tF_IDFVectorDocument[document][token]
                # Taking negative value to ease decreasing order sorting
                document_vector.append((-dot_product, document))
            document_vector.sort()

            # Write output to file
            outputData += str(query) + " : "
            for document in document_vector[:50]:
                outputData += str(document[1]) + " "
            outputData += "\n"

        # Open the file to write the data
        with open(rankedDocumentsOutputFileName , 'w') as outputFile:
            outputFile.write(outputData)
            print("Output file generated " + rankedDocumentsOutputFileName)

    # Function to display inverted index
    def displayInvertedIndex(self):
        print("Token:\tPostings List:")
        for token, postingsList in self.invertedIndex.items():
            print(token, end = "\t")
            print(postingsList)
            print("------------------------------------------------------------------------------------")

    # Function to display Id-document text map
    def displayIdDocumentMap(self):
        print("Document ID:\tDocument Text:")
        for docId, docText in self.iDDocumentMap.items():
            print(docId, end ="\t")
            print(docText)
            print("-------------------------------------------------------------------------------------")

    def writeDataToGivenFile(self, fileName, data):
        np.savetxt(fileName, data, fmt='%s')

# taking input file name from command line
if(len(sys.argv) != 4):
    print("Input format : <program name> <document input file name> <query input file name> <inverted index input file name>")
    sys.exit("Input format not correct")
inputFileDocument = str(sys.argv[1])
inputFileQuery = str(sys.argv[2])
inputFileInvertedIndex = str(sys.argv[3])

tF_IDFVectorizationAndEvaluation = TF_IDFVectorizationAndEvaluation(inputFileInvertedIndex, inputFileDocument, inputFileQuery)

# Create document frequency map
tF_IDFVectorizationAndEvaluation.loadInvertedIndex()
tF_IDFVectorizationAndEvaluation.getDocumentFrequency()

# Create processed documents
tF_IDFVectorizationAndEvaluation.createProcessedDocuments('document')
tF_IDFVectorizationAndEvaluation.createIdDocumentMap()
tF_IDFVectorizationAndEvaluation.createPreProcessedDocuments()

# Create processed queries
tF_IDFVectorizationAndEvaluation.createProcessedDocuments('query')
tF_IDFVectorizationAndEvaluation.createIdDocumentMap()
tF_IDFVectorizationAndEvaluation.createPreProcessedQueries()

# Create inverse document frequency map
tF_IDFVectorizationAndEvaluation.createInverseDocumentFrequency()

# Create TF vector for documents and queries
tF_IDFVectorizationAndEvaluation.createTFVectorDocument()
tF_IDFVectorizationAndEvaluation.createTFVectorQuery()

# Create TF-IDF vector for documents and queries in lnc.ltc format
tF_IDFVectorizationAndEvaluation.createTF_IDFVectorDocument_lncltc()

# Create TF-IDF vector for documents and queries in lnc.Ltc format
tF_IDFVectorizationAndEvaluation.createTF_IDFVectorDocument_lncLtc()

# Create TF-IDF vector for documents and queries in ancapc format
tF_IDFVectorizationAndEvaluation.createTF_IDFVectorDocument_ancapc()

# Rank documents based on cosine similarity for lnc.ltc format
tF_IDFVectorizationAndEvaluation.rankDocuments(tF_IDFVectorizationAndEvaluation.tF_IDFVectorDocument_lncltc, tF_IDFVectorizationAndEvaluation.tF_IDFVectorQuery_lncltc, TF_IDFVectorizationAndEvaluation.FILE_RANKED_LIST_lnc_ltd)

# Rank documents based on cosine similarity for Lnc.Lpc format
tF_IDFVectorizationAndEvaluation.rankDocuments(tF_IDFVectorizationAndEvaluation.tF_IDFVectorDocument_lncLtc, tF_IDFVectorizationAndEvaluation.tF_IDFVectorQuery_lncLtc, TF_IDFVectorizationAndEvaluation.FILE_RANKED_LIST_lnc_Ltc)

# Rank documents based on cosine similarity for ancapc format
tF_IDFVectorizationAndEvaluation.rankDocuments(tF_IDFVectorizationAndEvaluation.tF_IDFVectorDocument_ancapc, tF_IDFVectorizationAndEvaluation.tF_IDFVectorQuery_ancapc, TF_IDFVectorizationAndEvaluation.FILE_RANKED_LIST_anc_apc)