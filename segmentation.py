import json
import os
import torch
from sklearn.metrics.pairwise import cosine_similarity
import torch.nn as nn
from torch.utils.data import DataLoader

from sentence_transformers import SentenceTransformer

sBertModel = SentenceTransformer('all-MiniLM-L6-v2')

def getDocEmbeddings(doc):
    embeddings = sBertModel.encode(doc, convert_to_tensor=True)
    return embeddings
    
    # Compare the similarity between the embeddings of the sentences in the document
def segmentParas(embeddings, threshold = 0.6):
    segments = []
    segments.append(0)

    numSent = embeddings.shape[0]

    for i in range(1, numSent):
        # Calculate the cosine similarity between the current sentence and the previous sentence
        similarity = cosine_similarity([embeddings[i-1]], [embeddings[i]])[0][0]
        if (similarity < threshold):
            segments.append(i)
        # print(similarity)

    # print("Segments:")
    # print(segments)

    return segments

def adjustBySegment(allDocs, segIndices, allTargetSpans):
    newAllDocs = []
    newAllTargetSegs = []

    for seg in range(len(segIndices)):
        newCurrDoc = []
        newCurrTargetSegs = []

        spanToSeg = []

        for i in range(len(segIndices[seg])-1):
            if (segIndices[seg][i+1] - segIndices[seg][i] == 1):
                newCurrDoc.append(allDocs[seg][i])
                spanToSeg.append(i)
            else:
                newCurrDoc.append(" ".join(allDocs[seg][segIndices[seg][i]:segIndices[seg][i+1]]))
                spanToSeg.extend([i] * (segIndices[seg][i+1] - segIndices[seg][i]))
        
        newCurrDoc.append(" ".join(allDocs[seg][segIndices[seg][len(segIndices[seg])-1]:]))
        spanToSeg.extend([i] * (len(allDocs[seg][segIndices[seg][len(segIndices[seg])-1]:])))

        newAllDocs.append(newCurrDoc)

        for i in range(len(allTargetSpans[seg])):
            tempTargets = []
            if allTargetSpans[seg][i] == []:
                newCurrTargetSegs.append([])
            else:
                prev = -1
                for span in allTargetSpans[seg][i]:
                    if spanToSeg[span] != prev:
                        tempTargets.append(spanToSeg[span])
                        prev = spanToSeg[span]
                newCurrTargetSegs.append(tempTargets)

        newAllTargetSegs.append(newCurrTargetSegs)

    return newAllDocs, newAllTargetSegs