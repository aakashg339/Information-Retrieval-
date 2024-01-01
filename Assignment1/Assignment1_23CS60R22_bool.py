import sys
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pickle

# class that uses the inverted index to get result as per query
class BooleanRetrivalImplementation:
    def __init__(self, invertedIndex, queryFile, outputFile):
        self.invertedIndexFile = invertedIndex
        self.queryFile = queryFile
        self.outputFile = outputFile
        self.queryResult = {}
        self.invertedIndex = {}
        self.processedQuery = ""
        self.queryResultText = ""
        self.defaultLemmatizer = WordNetLemmatizer()
        self.defaultStopWords = set(stopwords.words('english'))

    # Function to get the inverted index from file
    def loadInvertedIndex(self):
        try:
            with open(self.invertedIndexFile, 'rb') as invertedIndexFile:
                self.invertedIndex = pickle.load(invertedIndexFile)
        except FileNotFoundError:
            sys.exit("File " + self.invertedIndexFile + " not found. Kindly check the file name")

    # Function to get the processed query from file
    def loadProcessesQuery(self):
        try:
            with open(self.queryFile, 'r') as queryFile:
                self.processedQuery = queryFile.readlines()
        except FileNotFoundError:
            sys.exit("File " + self.queryFile + " not found. Kindly check the file name")

    # Function to get words out of the query
    def getQueryWords(self, query):
        queryWords = query.split();
        queryWordsSpaceRemoved = []
        for queryWord in queryWords:
            queryWordsSpaceRemoved.append(queryWord.strip())
        return queryWordsSpaceRemoved

    # Function to merge to posting list
    def mergeTwoIdList(self, list1, list2):
        i=0
        j=0
        mergedList = []
        while i<len(list1) and j<len(list2):
            if(list1[i] == list2[j]):
                mergedList.append(list1[i])
                i+=1
                j+=1
            elif(list1[i] < list2[j]):
                i+=1
            else:
                j+=1
        return mergedList

    # Function to get document ids for a given query. It is helper method for processQueries()
    def getDocumentIdsForGivenQueryWords(self, queryWords):
        documentIds = None
        for word in queryWords:
            if word in self.invertedIndex:
                postingsList = self.invertedIndex[word]
                if documentIds is None:
                    documentIds = postingsList
                else:
                    documentIds = self.mergeTwoIdList(documentIds, postingsList)
        return documentIds
    
    # Function to get document ids for every query using the function getDocumentIdsForGivenQueryWords()
    def processQueries(self):
        for query in self.processedQuery:
            idAndQuery = query.strip().split('\t')
            queryId = idAndQuery[0]
            queryWords = self.getQueryWords(idAndQuery[1])

            documentIdsForQuery = self.getDocumentIdsForGivenQueryWords(queryWords)

            documentIdsForQueryForQueryInString = []
            for documentIdForQuery in documentIdsForQuery:
                documentIdsForQueryForQueryInString.append(str(documentIdForQuery))

            self.queryResult[queryId] = documentIdsForQuery
            str1 = " "
            self.queryResultText += str(queryId) + " : " + str1.join(documentIdsForQueryForQueryInString) + "\n"

    # Function to save the result of the query
    def saveQueryResults(self):
        with open(self.outputFile, 'w') as output_file:
            output_file.write(self.queryResultText)
            print("Boolean retrival done. Output written to file " + self.outputFile)

    # Function to display inverted index
    def displayInvertedIndex(self):
        print("Token:\tPostings List:")
        for token, postingsList in self.invertedIndex.items():
            print(token, end = "\t")
            print(postingsList)
            print("------------------------------------------------------------------------------------")

    # Function to display the processed query 
    def displayProcessedQuery(self):
        print(self.processedQuery)

    # Function to display query result
    def displayQueryResults(self):
        print("Query ID:\tDocuments List :")
        for queryId, documentIdsForQuery in self.queryResult.items():
            print(queryId, end = "\t")
            if documentIdsForQuery:
                print(documentIdsForQuery)
            else:
                print("None")
            print("------------------------------------------------------------------------------------")

# Files names for input and output
# taking input file name from command line
if(len(sys.argv) != 3):
    print("Input format : <program name> <Inverted index input file name with path> <query input file with path>")
    sys.exit("Input format not correct")
inputFileForInvertedIndex = str(sys.argv[1])
inputFileForProcessedQuery = str(sys.argv[2])
#inputFileForInvertedIndex = 'model_queries_23CS60R22.bin'
#inputFileForProcessedQuery = 'queries_23CS60R22.txt'
outputFile = 'Assignment1_23CS60R22_results.txt'
booleanRetrivalImplementation = BooleanRetrivalImplementation(inputFileForInvertedIndex, inputFileForProcessedQuery, outputFile)

booleanRetrivalImplementation.loadInvertedIndex()
#booleanRetrivalImplementation.displayInvertedIndex()
booleanRetrivalImplementation.loadProcessesQuery()
#booleanRetrivalImplementation.displayProcessedQuery()
booleanRetrivalImplementation.processQueries()
#booleanRetrivalImplementation.displayQueryResults()
booleanRetrivalImplementation.saveQueryResults()
