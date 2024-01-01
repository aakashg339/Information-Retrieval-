import re
import sys
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pickle

# Class to build inverted index from the given data
class InvertedIndexerForDocuments:
    def __init__(self, inputFile, outputFile):
        self.inputFile = inputFile
        self.outputFile = outputFile
        self.processedDocuments = []
        self.iDDocumentMap = {}
        self.invertedIndex = {}
        self.defaultLemmatizer = WordNetLemmatizer()
        self.defaultStopWords = set(stopwords.words('english'))

    # Function to read data from input file
    def createProcessedDocuments(self):
        # Reading from the given input file
        try:
            with open(self.inputFile, 'r') as file:
                inputFileData = file.read()
        except FileNotFoundError:
            sys.exit("File " + self.inputFile + " not found. Kindly check the file name")

        # Spliting the data based on .I
        documents = re.split(r'(\.I \d+)', inputFileData)

        # Remove extra spaces
        for document in documents:
            strippedDocument = document.strip()
            if strippedDocument:
                self.processedDocuments.append(strippedDocument)
    
    # Fuction to create the id and text data map
    def createIdDocumentMap(self):
        documentId = None
        documentText = None
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

    # Helper method of createInvertedIndex() to get lemmatized tokens
    def processDocument(self, documentText):
        wordTokens = word_tokenize(documentText)

        wordTokensPunctuationRemoved = self.removePunctuationConvertLower(wordTokens)
        
        wordTokensStopWordsRemoved = self.removeStopWords(wordTokensPunctuationRemoved)
        
        lemmatizedTokens = self.lemmatizeWords(wordTokensStopWordsRemoved)

        return lemmatizedTokens

    # Function to create inverted index
    def createInvertedIndex(self):
        for documentId, documentText in self.iDDocumentMap.items():
            # Get lemmatized words for each text data
            processedDocumentText = self.processDocument(documentText)

            # Creating the inverted index
            for wordToken in processedDocumentText:
                if wordToken not in self.invertedIndex:
                    self.invertedIndex[wordToken] = [documentId]
                else:
                    if documentId not in self.invertedIndex[wordToken]:
                        self.invertedIndex[wordToken].append(documentId)
        
        # Sorting the inverted index based on keys
        sortedItem = sorted(self.invertedIndex.items())
        self.invertedIndex = dict(sortedItem)

        # Sorting the postings list for each word in inverted index
        for token, postingsList in self.invertedIndex.items():
            self.invertedIndex[token] = sorted(postingsList)

    # Function to save inverted index in a .bin file
    def saveInvertedIndex(self):
        with open(self.outputFile, "wb") as outputFile:
            pickle.dump(self.invertedIndex, outputFile)
            print("Inverted Index created. Output written to file " + self.outputFile)

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

# taking input file name from command line
if(len(sys.argv) != 2):
    print("Input format : <program name> <input file name>")
    sys.exit("Input format not correct")
inputFilename = str(sys.argv[1])
#inputFilename = '/home/aakashg339/Documents/PythonWorkspace/python_venv_/CollegeAssignment/IR/Assignment1/cran.all.1400'
outputFilename = 'model_queries_23CS60R22.bin'

# Start with processing the document and creating the inverted index
invertedIndexerForDocuments = InvertedIndexerForDocuments(inputFilename, outputFilename)
invertedIndexerForDocuments.createProcessedDocuments()
invertedIndexerForDocuments.createIdDocumentMap()
#invertedIndexerForDocuments.displayIdDocumentMap()
invertedIndexerForDocuments.createInvertedIndex()
#invertedIndexerForDocuments.displayInvertedIndex()
invertedIndexerForDocuments.saveInvertedIndex()
