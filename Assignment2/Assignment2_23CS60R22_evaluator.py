import sys
import numpy as np

class Evaluator:

    # Defining constants
    FILE_EVALUATION_OUTPUT = "Assignment2_23CS60R22_metrics_"

    def __init__(self, goldStandardRankedListFileName, rankedListToBeEvaluatedFileName, fileUsed):
        self.goldStandardRankedListFileName = goldStandardRankedListFileName
        self.rankedListToBeEvaluatedFileName = rankedListToBeEvaluatedFileName
        self.outputFile = Evaluator.FILE_EVALUATION_OUTPUT + fileUsed + ".txt"
        self.goldStandardRanked_QueryId_DocumentIdAndScoreList_Map = {}
        self.rankedListToBeEvaluated_QueryId_DocumentList_Map = {}
        self.relevantDocumentsGoldStandard_QueryId_Documents_Map = {}
        self.averagePrecisionAt10_QueryId_AveragePrecision_Map = {}
        self.averagePrecisionAt20_QueryId_AveragePrecision_Map = {}
        self.queryIds_DocumentIdAndRelevanceScoreList_Map = {}
        self.QueryId_NDCGAt10_Map = {}
        self.QueryId_NDCGAt20_Map = {}
        self.indexQueryIdMap = {}

    def readGoldStandardRankedList(self):
        with open(self.goldStandardRankedListFileName, 'r') as f:
            for line in f:
                data = line.strip().split()
                queryId = data[0]
                documentId = data[1]
                relevanceScore = int(data[2])
                if queryId not in self.goldStandardRanked_QueryId_DocumentIdAndScoreList_Map:
                    self.goldStandardRanked_QueryId_DocumentIdAndScoreList_Map[queryId] = []
                self.goldStandardRanked_QueryId_DocumentIdAndScoreList_Map[queryId].append([documentId, relevanceScore])

        # print(self.goldStandardRanked_QueryId_DocumentIdAndScoreList_Map)
    
    def readRankedLists(self):
        with open(self.rankedListToBeEvaluatedFileName, 'r') as f:
            for index, line in enumerate(f):
                data = line.strip().split(':')
                queryId = data[0]
                documentIds = data[1].strip().split()
                self.rankedListToBeEvaluated_QueryId_DocumentList_Map[str(index+1)] = documentIds
                self.indexQueryIdMap[str(index+1)] = queryId

        
        # print(self.rankedListToBeEvaluated_QueryId_DocumentList_Map)
    
    # Funtion to get the relevant documents from the gold standard ranked list
    def buildRelevantDocumentsList(self):
        for queryId in self.goldStandardRanked_QueryId_DocumentIdAndScoreList_Map:
            for documentId, relevanceScore in self.goldStandardRanked_QueryId_DocumentIdAndScoreList_Map[queryId]:
                if queryId not in self.relevantDocumentsGoldStandard_QueryId_Documents_Map:
                    self.relevantDocumentsGoldStandard_QueryId_Documents_Map[queryId] = []
                if int(relevanceScore) > 0:
                    self.relevantDocumentsGoldStandard_QueryId_Documents_Map[queryId].append(documentId)

        #print(self.relevantDocumentsGoldStandard_QueryId_Documents_Map)
    
    # Function to calculate average precision @ k for the given query
    def calculateAveragePrecisionForEachQuery(self, queryId, k):
        relevantRetrieved = 0
        relevantRetrievedList = []
        precision = 0

        if queryId not in self.relevantDocumentsGoldStandard_QueryId_Documents_Map:
            return 0

        for i in range(k):
            if  self.rankedListToBeEvaluated_QueryId_DocumentList_Map[queryId][i] in self.relevantDocumentsGoldStandard_QueryId_Documents_Map[queryId]:
                relevantRetrieved += 1
                relevantRetrievedList.append(self.rankedListToBeEvaluated_QueryId_DocumentList_Map[queryId][i])
                precision += relevantRetrieved / (i + 1)
        if relevantRetrieved == 0:
            return 0
        precision /= relevantRetrieved
        return precision
    
    # Function to calculate precision @ k for each query by using function calculateAveragePrecisionForEachQuery
    def calculateAveragePrecision(self):
        for queryId in self.rankedListToBeEvaluated_QueryId_DocumentList_Map:
            self.averagePrecisionAt10_QueryId_AveragePrecision_Map[queryId] = self.calculateAveragePrecisionForEachQuery(queryId, 10)
            self.averagePrecisionAt20_QueryId_AveragePrecision_Map[queryId] = self.calculateAveragePrecisionForEachQuery(queryId, 20)
        
        # print(self.averagePrecisionAt10_QueryId_AveragePrecision_Map)
        # print(self.averagePrecisionAt20_QueryId_AveragePrecision_Map)

    # Function to extract document id and relevance score from gold standard ranked list, for all query id. 
    def extractDocumentIdAndRelevanceScoreFromGoldStandardRankedList(self):
        # Each key of the map contains a map of document id and relevance score
        for queryId in self.goldStandardRanked_QueryId_DocumentIdAndScoreList_Map:
            documentIdAndRelevanceScoreMap = {}
            for documentId, score in self.goldStandardRanked_QueryId_DocumentIdAndScoreList_Map[queryId]:
                documentIdAndRelevanceScoreMap[documentId] = score
            self.queryIds_DocumentIdAndRelevanceScoreList_Map[queryId] = documentIdAndRelevanceScoreMap

    # Function to calculate Normalized Discounted Cumulative Gain @ k for a given query
    def calculateNDCGForEachQuery(self, queryId, k):
        if queryId not in self.relevantDocumentsGoldStandard_QueryId_Documents_Map:
            return 0
        
        # Creating a 2D array of size 6 * k, where each row contains below values
        # 1. Document Ids as per there relating ordering in the gold standard ranked list
        # 2. Relevance score of the document
        # 3. DCG value
        # 4. Document Ids as per there relating ordering in the ranked list to be evaluated
        # 5. Relevance score of the document
        # 6. DCG value
        calculatingMatrix = np.zeros((6, k))
        documentIdAndRelevanceScoreMap = self.queryIds_DocumentIdAndRelevanceScoreList_Map[queryId]

        # Checking if documentIdAndRelevanceScoreMap is empty
        if not documentIdAndRelevanceScoreMap:
            return 0

        # Populating the matrix with values
        for i in range(k):
            # For gold standard ranked list
            if i < len(self.relevantDocumentsGoldStandard_QueryId_Documents_Map[queryId]):
                documentId = self.relevantDocumentsGoldStandard_QueryId_Documents_Map[queryId][i]
                relevanceScore = documentIdAndRelevanceScoreMap[documentId]
                if relevanceScore < 0:
                    relevanceScore = 0
                calculatingMatrix[0][i] = documentId
                calculatingMatrix[1][i] = relevanceScore
                dcgValue = relevanceScore
                if i > 0:
                    dcgValue = dcgValue / np.log2(i + 1)
                calculatingMatrix[2][i] = dcgValue
            # For ranked list to be evaluated
            if i < len(self.rankedListToBeEvaluated_QueryId_DocumentList_Map[queryId]):
                documentId = self.rankedListToBeEvaluated_QueryId_DocumentList_Map[queryId][i]
                if documentId in documentIdAndRelevanceScoreMap:
                    relevanceScore = documentIdAndRelevanceScoreMap[documentId]
                else:
                    relevanceScore = 0
                if relevanceScore < 0:
                    relevanceScore = 0
                calculatingMatrix[3][i] = documentId
                calculatingMatrix[4][i] = relevanceScore
                dcgValue = relevanceScore
                if i > 0:
                    dcgValue = dcgValue / np.log2(i + 1)
                calculatingMatrix[5][i] = dcgValue
            
        DCGIdeal = np.sum(calculatingMatrix[2])
        DCGActual = np.sum(calculatingMatrix[5])
        if DCGIdeal == 0:
            return 0
        nDCG = DCGActual / DCGIdeal

        return nDCG
    
    # Function to calculate Normalized Discounted Cumulative Gain @ k for all queries
    def calculateNDCG(self):
        for queryId in self.rankedListToBeEvaluated_QueryId_DocumentList_Map:
            self.QueryId_NDCGAt10_Map[queryId] = self.calculateNDCGForEachQuery(queryId, 10)
            self.QueryId_NDCGAt20_Map[queryId] = self.calculateNDCGForEachQuery(queryId, 20)
    
    # Function to calculate mean average precision @ 10
    def calculateMeanAveragePrecisionAt10(self):
        sumAveragePrecision = 0
        for queryId in self.averagePrecisionAt10_QueryId_AveragePrecision_Map:
            sumAveragePrecision += self.averagePrecisionAt10_QueryId_AveragePrecision_Map[queryId]
        meanAveragePrecision = sumAveragePrecision / len(self.averagePrecisionAt10_QueryId_AveragePrecision_Map)
        return meanAveragePrecision
    
    # Function to calculate mean average precision @ 20
    def calculateMeanAveragePrecisionAt20(self):
        sumAveragePrecision = 0
        for queryId in self.averagePrecisionAt20_QueryId_AveragePrecision_Map:
            sumAveragePrecision += self.averagePrecisionAt20_QueryId_AveragePrecision_Map[queryId]
        meanAveragePrecision = sumAveragePrecision / len(self.averagePrecisionAt20_QueryId_AveragePrecision_Map)
        return meanAveragePrecision
    
    # Function to calculate mean NDCG @ 10
    def calculateMeanNDCGAt10(self):
        sumNDCG = 0
        for queryId in self.QueryId_NDCGAt10_Map:
            sumNDCG += self.QueryId_NDCGAt10_Map[queryId]
        meanNDCG = sumNDCG / len(self.QueryId_NDCGAt10_Map)
        return meanNDCG

    # Function to calculate mean NDCG @ 20
    def calculateMeanNDCGAt20(self):
        sumNDCG = 0
        for queryId in self.QueryId_NDCGAt20_Map:
            sumNDCG += self.QueryId_NDCGAt20_Map[queryId]
        meanNDCG = sumNDCG / len(self.QueryId_NDCGAt20_Map)
        return meanNDCG
    
    # Function to write the output to a file
    def writeOutputToFile(self):
        output = ""
        
        with open(self.outputFile, 'w') as f:
            # Writing average precison @ 10
            output += "Average Precision @ 10\n"
            for queryId in self.averagePrecisionAt10_QueryId_AveragePrecision_Map:
                output += self.indexQueryIdMap[queryId] + " " + str(self.averagePrecisionAt10_QueryId_AveragePrecision_Map[queryId]) + "\n"
            output += "\n"
            
            # Writing average precison @ 20
            output += "Average Precision @ 20\n"
            for queryId in self.averagePrecisionAt20_QueryId_AveragePrecision_Map:
                output += self.indexQueryIdMap[queryId] + " " + str(self.averagePrecisionAt20_QueryId_AveragePrecision_Map[queryId]) + "\n"
            output += "\n"

            # Writing NDCG@10
            output += "NDCG @ 10\n"
            for queryId in self.QueryId_NDCGAt10_Map:
                output += self.indexQueryIdMap[queryId] + " " + str(self.QueryId_NDCGAt10_Map[queryId]) + "\n"
            output += "\n"

            # Writing NDCG@20
            output += "NDCG @ 20\n"
            for queryId in self.QueryId_NDCGAt20_Map:
                output += self.indexQueryIdMap[queryId] + " " + str(self.QueryId_NDCGAt20_Map[queryId]) + "\n"
            output += "\n"

            # Writing mean average precision @ 10
            output += "Mean Average Precision @ 10\n"
            output += str(self.calculateMeanAveragePrecisionAt10()) + "\n"
            output += "\n"

            # Writing mean average precision @ 20
            output += "Mean Average Precision @ 20\n"
            output += str(self.calculateMeanAveragePrecisionAt20()) + "\n"
            output += "\n"

            # Writing mean NDCG @ 10
            output += "Mean NDCG @ 10\n"
            output += str(self.calculateMeanNDCGAt10()) + "\n"
            output += "\n"

            # Writing mean NDCG @ 20
            output += "Mean NDCG @ 20\n"
            output += str(self.calculateMeanNDCGAt20()) + "\n"

            f.write(output)

# taking input file name from command line
if(len(sys.argv) != 3):
    print("Input format : <program name> <gold standard ranked list> <rank list to be evaluated>")
    sys.exit("Input format not correct")
inputFileGoldStandardRankedList = str(sys.argv[1].strip())
inputFileRankedListToBeEvaluated = str(sys.argv[2].strip())
indexLastUnderscore = inputFileRankedListToBeEvaluated.rfind('_')
indexDot = inputFileRankedListToBeEvaluated.rfind('.')
fileUsed = inputFileRankedListToBeEvaluated[indexLastUnderscore + 1:indexDot]

evaluator = Evaluator(inputFileGoldStandardRankedList, inputFileRankedListToBeEvaluated, fileUsed)
evaluator.readGoldStandardRankedList()
evaluator.readRankedLists()
evaluator.buildRelevantDocumentsList()
evaluator.calculateAveragePrecision()

evaluator.extractDocumentIdAndRelevanceScoreFromGoldStandardRankedList()
evaluator.calculateNDCG()

evaluator.writeOutputToFile()
