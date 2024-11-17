# Load and parse the json file:
import json
import os
import torch
from sklearn.metrics.pairwise import cosine_similarity
import torch.nn as nn
from torch.utils.data import DataLoader
# from sentence_transformers import SentenceTransformer

from datasett import DataSetGen, loadDocs, relevanceLabelling, prepHypoClassiData, testHypoClassiData
# from segmentation import getDocEmbeddings, segmentParas, adjustBySegment
from hardcode import targetChoices, hypoLabels
from relevanceClassi import trainRelevance, getRelevanceLabels
from hypoClassi import train_hypothesis_classification, classify_hypothesis, testHypoClassification
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

################

BatchSize = 16

################
    
if __name__ == '__main__':

    config = {
        "trainMode": False,
        "segmentation": False,
        "task1": False,
        "task2": False,
        'humanMode': True
    }

    filePathTrain = "./data/train.json"
    filePathTest = "./data/test.json"
    filePathDev = "./data/dev.json"

    if config["trainMode"]:

        allDocs, allTargetChoice, allTargetSpans, hypothesis = loadDocs(filePathTrain, hypoLabels)
        allDocsTest, allTargetChoiceTest, allTargetSpansTest, hypothesisTest = loadDocs(filePathTest, hypoLabels)
        allDocsDev, allTargetChoiceDev, allTargetSpansDev, hypothesisDev = loadDocs(filePathDev, hypoLabels)

        """
        allDocs: It is a list where each entry is for one doc. Each entry itself is a list of strings (each string being a span)
        allTargetChoice: It is a list of lists. Each list corresponds to a doc. Each entry of the list is the target choice for that hypothesis. totally 17 choices in the inner list (as 17 hyps)
        stored as 0, 1 or 2.
        allTargetSpans: It is a list of list of lists. Each list corresponds to a doc. Each entry of the list is a list of spans for that hypothesis. totally 17 lists in the inner list (as 17 hyps)
        hypothesis: It is a list of 17 hypothesis (the text itself) (no diff btwn hypothesis, hypothesisTest and Dev. same only)
        """

        # # If You do NOT want to perform segmentation, comment the following lines
        # #########################
        # # segmenter = segmentation(all)    
        # if config["segmentation"]:
        #     sBertModel = SentenceTransformer('all-MiniLM-L6-v2')
        #     segmentThreshold = 0.6
        #     segIndices = []
        #     # Getting index of segments
        #     for doc in allDocs:
        #         embeddings = getDocEmbeddings(doc)
        #         segIndices.append(segmentParas(embeddings, segmentThreshold))

        #     # Now spans are no more the original spans. They are the new segments. Both in the document and in the target
        #     allDocs, allTargetSpans = adjustBySegment(allDocs, segIndices, allTargetSpans)
        # #########################


        """
        # STEPS:
        # TASK 1: IDENTIFY THE RELEVANT SEGMENTS FOR EACH HYPOTHESIS
        # TASK 2: CLASSIFY THE HYPOTHESIS BASED ON RELEVANT SEGMENTS
        """

        # Task 1: Relevance Classification:
        if config["task1"]:
            trainInp, trainOut = relevanceLabelling(allDocs, allTargetChoice, allTargetSpans, hypothesis)
            devInp, devOut = relevanceLabelling(allDocsDev, allTargetChoiceDev, allTargetSpansDev, hypothesisDev)
            """
            return values of relevanceLabelling:
            Inp: list of "hypo" + " " + "segment"
            Out: list of 1 or 0 (relevant or not)
            """
            trainRelevance(trainInp, trainOut, devInp, devOut)

        
        # # Task 2: Hypothesis Classification:
        if config["task2"]:
            classiTrainInp, classiTrainOut = prepHypoClassiData(allDocs, allTargetChoice, allTargetSpans, hypothesis)
            classiDevInp, classiDevOut = prepHypoClassiData(allDocsDev, allTargetChoiceDev, allTargetSpansDev, hypothesisDev)

            train_hypothesis_classification(classiTrainInp, classiTrainOut, classiDevInp, classiDevOut)

    elif config['humanMode']:
        # This is just testing on 3 examples so we can see the output  
        
        relClassiPath = "./relevance_classifier"
        hypoClassiPath = "./hypothesis_classifier"

        numDocs = 10
        print("Loading 3 docs from the test data")
        allDocsTest, allTargetChoiceTest, allTargetSpansTest, hypothesisTest = loadDocs(filePathTest, hypoLabels)
        allDocsTest = allDocsTest[:numDocs]
        allTargetChoiceTest = allTargetChoiceTest[:numDocs]
        allTargetSpansTest = allTargetSpansTest[:numDocs]
        
        for i in range(numDocs):
            print("Doc: ", i)
            print("Document Spans: ", allDocsTest[i][0:3], "...")
            print("Hypothesis Target Choices: ", allTargetChoiceTest[i])
            print("Hypothesis Target Spans: ", allTargetSpansTest[i][0:3], "...")
            print("Num of Spans: ", len(allDocsTest[i]))
            print()

        testInp, testRelTarget = relevanceLabelling(allDocsTest, allTargetChoiceTest, allTargetSpansTest, hypothesisTest)

        print("Performing relevance classification")
        testRelPred = getRelevanceLabels(testInp, relClassiPath)
        print("Accuracy: ", accuracy_score(testRelTarget, testRelPred))
        print("Relevance classification done")

        testHypoInp, noMentionIndices = testHypoClassiData(allDocsTest, testRelPred, hypothesisTest)
        print("Performing hypothesis classification")
        # print("No Mention Indices: ")
        # print(noMentionIndices)
        testHypoPred = testHypoClassification(testHypoInp, hypoClassiPath)

        testHypoTarget = [lbl for doc in allTargetChoiceTest for lbl in doc]
        print("Target:")
        print(testHypoTarget)
        print("Predicted:")
        print(testHypoPred)
        print("Accuracy: ", accuracy_score(testHypoTarget, testHypoPred))
        
        # for indx in noMentionIndices:
        #     testHypoPred[indx] = targetChoices["NotMentioned"]
        # print("After Modd")
        # print("ModPredicted:")
        # print(testHypoPred)
        print("Hypothesis classification done")

    else:
        """
        # Inference mode
        # Steps:
        # 1. Load the trained models
        # 2. Load the test data
        # 3. Perform segmentation
        # 4. Perform relevance classification
        # 5. Perform hypothesis classification
        """

        # Load the trained models
        relClassiPath = "./relevance_classifier"
        hypoClassiPath = "./hypothesis_classifier"

        # Load the test data
        print("Loading the test data")
        allDocsTest, allTargetChoiceTest, allTargetSpansTest, hypothesisTest = loadDocs(filePathTest, hypoLabels)
        print("Test data loaded")

        # # Perform segmentation
        # if config["segmentation"]:
        #     print("Performing segmentation")
        #     sBertModel = SentenceTransformer('all-MiniLM-L6-v2')
        #     segmentThreshold = 0.6
        #     segIndices = []
        #     # Getting index of segments
        #     for doc in allDocsTest:
        #         embeddings = getDocEmbeddings(doc)
        #         segIndices.append(segmentParas(embeddings, segmentThreshold))

        #     # Now spans are no more the original spans. They are the new segments. Both in the document and in the target
        #     allDocsTest, allTargetSpansTest = adjustBySegment(allDocsTest, segIndices, allTargetSpansTest)

        # Perform relevance classification
        testInp, testRelTarget = relevanceLabelling(allDocsTest, allTargetChoiceTest, allTargetSpansTest, hypothesisTest)
        print("Performing relevance classification")
        testRelPred = getRelevanceLabels(testInp, relClassiPath)
        print("Relevance classification done")

        # Perform hypothesis classification
        testHypoInp, noMentionIndices = testHypoClassiData(allDocsTest, testRelPred, hypothesisTest)
        print("Performing hypothesis classification")
        testHypoPred = testHypoClassification(testHypoInp, hypoClassiPath)
        print("Hypothesis classification done")

        # Evaluation:
        print("Relevance Prediction:")
        # My target is testRelTarget and my prediction is testRelPred now print the score
        RelAccuracy = accuracy_score(testRelTarget, testRelPred)
        print("Accuracy: ", RelAccuracy)

        print("Hypothesis Prediction:")
        testHypoTarget = [lbl for doc in allTargetChoiceTest for lbl in doc]

        HypoAccuracy = accuracy_score(testHypoTarget, testHypoPred)
        print("Accuracy: ", HypoAccuracy)