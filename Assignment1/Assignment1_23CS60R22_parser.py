import re
import sys
import string
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Class to read data from input file and create processed queries
class QueryProcessor:
    def __init__(self, input_file, output_file):
        self.inputFile = input_file
        self.outputFile = output_file
        self.processedDocuments = []
        self.iDDocumentMap = {}
        self.processedIDDocumentMap = {}
        self.processedQuery = ""
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

        # Spliting the file based on .I
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
                documentText = processedDocument[processedDocument.find('.W')+2:] # find('.W')+2 is included to not include .W in result
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

   # Function to process query document 
    def createProcessedIDDocumentMap(self):
        for documentId, documentText in self.iDDocumentMap.items():
            # Get lemmatized words for each query
            processedDocumentText = self.processDocument(documentText)
            str1 = " "

            # Creating queryId-query map
            self.processedIDDocumentMap[documentId] = str1.join(processedDocumentText)

            # Creating the string data that needs to be written to file
            self.processedQuery += str(documentId) + "\t" + self.processedIDDocumentMap[documentId] + "\n"
            
    # Function to save processed query in a .txt file
    def saveProcessedQuery(self):
        with open(self.outputFile, 'w') as outputFile:
            outputFile.write(self.processedQuery)
            print("Query processed. Output written to file " + self.outputFile)

    # Function to display processed QueryId-Query map
    def displayIdDocumentMap(self):
        print("Document ID:\tDocument Text:")
        for docId, docText in self.iDDocumentMap.items():
            print(docId, end ="\t")
            print(docText)
            print("-------------------------------------------------------------------------------------")

    # Function to display processed QueryId-Processed Query map
    def displayProcessedIdDocumentMap(self):
        print("Document ID:\tDocument Text:")
        for docId, docText in self.processedIDDocumentMap.items():
            print(docId, end ="\t")
            print(docText)
            print("-------------------------------------------------------------------------------------")

# taking input file name from command line
if(len(sys.argv) != 2):
    print("Input format : <program name> <input file name with path>")
    sys.exit("Input format not correct")
inputFilename = str(sys.argv[1])
#inputFilename = '/home/aakashg339/Documents/PythonWorkspace/python_venv_/Practice/IR/Assignment1/cran.qry'
outputFilename = 'queries_23CS60R22.txt'

# Start with processing the document and creating processed query
queryParser = QueryProcessor(inputFilename, outputFilename)
queryParser.createProcessedDocuments()
queryParser.createIdDocumentMap()
#queryParser.displayIdDocumentMap()
queryParser.createProcessedIDDocumentMap()
#queryParser.displayProcessedIdDocumentMap()
queryParser.saveProcessedQuery()
