# Load and parse the json file:
import json
import os
import torch
from sklearn.metrics.pairwise import cosine_similarity
import torch.nn as nn
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer

from datasett import DataSetGen, loadDocs, relevanceLabelling, prepHypoClassiData
from segmentation import getDocEmbeddings, segmentParas, adjustBySegment
from hardcode import targetChoices, hypoLabels
from relevanceClassi import trainRelevance, getRelevanceLabels
from hypoClassi import train_hypothesis_classification, classify_hypothesis


################

BatchSize = 16

################
    
if __name__ == '__main__':
    filePathTrain = "./data/train.json"
    filePathTest = "./data/test.json"
    filePathDev = "./data/dev.json"

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
    # sBertModel = SentenceTransformer('all-MiniLM-L6-v2')
    # segmentThreshold = 0.6
    # segIndices = []
    # # Getting index of segments
    # for doc in allDocs:
    #     embeddings = getDocEmbeddings(doc)
    #     segIndices.append(segmentParas(embeddings, segmentThreshold))

    # # Now spans are no more the original spans. They are the new segments. Both in the document and in the target
    # allDocs, allTargetSpans = adjustBySegment(allDocs, segIndices, allTargetSpans)
    # #########################


    """
    # STEPS:
    # TASK 1: IDENTIFY THE RELEVANT SEGMENTS FOR EACH HYPOTHESIS
    # TASK 2: CLASSIFY THE HYPOTHESIS BASED ON RELEVANT SEGMENTS
    """

    # Task 1: Relevance Classification:
    trainInp, trainOut = relevanceLabelling(allDocs, allTargetChoice, allTargetSpans, hypothesis)
    testInp, testOut = relevanceLabelling(allDocsTest, allTargetChoiceTest, allTargetSpansTest, hypothesisTest)
    devInp, devOut = relevanceLabelling(allDocsDev, allTargetChoiceDev, allTargetSpansDev, hypothesisDev)

    """
    return values of relevanceLabelling:
    Inp: list of "hypo" + " " + "segment"
    Out: list of 1 or 0 (relevant or not)
    """

    trainRelevance(trainInp, trainOut, devInp, devOut)
    relClassiPath = "./relevance_classifier" 
    # getRelevanceLabels(testInp, relClassiPath)


    # Task 2: Hypothesis Classification:
    classiTrainInp, classiTrainOut = prepHypoClassiData(allDocs, allTargetChoice, allTargetSpans, hypothesis)
    classiDevInp, classiDevOut = prepHypoClassiData(allDocsDev, allTargetChoiceDev, allTargetSpansDev, hypothesisDev)

    train_hypothesis_classification(classiTrainInp, classiTrainOut, classiDevInp, classiDevOut)