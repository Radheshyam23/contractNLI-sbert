# Load and parse the json file:
import json
import os
import torch
from sklearn.metrics.pairwise import cosine_similarity
import torch.nn as nn
from torch.utils.data import DataLoader

from sentence_transformers import SentenceTransformer


################
targetChoices = {
    "Contradiction": 0,
    "Entailment": 1,
    "NotMentioned": 2 
}
BatchSize = 16

################


def loadDocs(filePath, hypoLabels):
    with open(filePath, 'r') as f:
        data = json.load(f)

    print("Number of documents in the dataset:"+str(len(data["documents"])))

    allDocs = []
    allTargetChoice = []
    allTargetSpans = []

    hypothesis = []

    for hypLabel in hypoLabels:
        hypothesis.append(data["labels"][hypLabel]["hypothesis"])

    for doc in data["documents"]:
        # Since not all spans are aligned with '\n', we are splitting by span instead of \n
        # allDocs.append(doc["text"].split("\n"))

        currDoc = []
        spans=doc["spans"]
        
        for span in spans:
            currDoc.append(doc["text"][span[0]:span[1]])

        allDocs.append(currDoc)

        # This will have 17 different hypothesis and their labels and spans. 
        # anno = doc["annotation_sets"][0]["annotations"]

        currChoices = []
        currSpans = []
        for hypLabel in hypoLabels:
            currChoices.append(targetChoices[doc["annotation_sets"][0]["annotations"][hypLabel]["choice"]])
            currSpans.append(doc["annotation_sets"][0]["annotations"][hypLabel]["spans"])

        allTargetChoice.append(currChoices)
        allTargetSpans.append(currSpans)

    return allDocs, allTargetChoice, allTargetSpans, hypothesis

# Gets the embeddings for all sentences in the document
def getDocEmbeddings(doc):
    embeddings = sBertModel.encode(doc, convert_to_tensor=True)
    return embeddings

# Segmentation
# Compare the similarity between the embeddings of the sentences in the document
def segmentParas(embeddings, doc, threshold = 0.6):
    segments = []
    segments.append(0)

    numSent = embeddings.shape[0]

    for i in range(1, numSent):
        # Calculate the cosine similarity between the current sentence and the previous sentence
        similarity = cosine_similarity([embeddings[i-1]], [embeddings[i]])[0][0]
        if (similarity < threshold):
            segments.append(i)
        print(similarity)

    print("Segments:")
    print(segments)

    return segments


class ClassificationModel(nn.Module):
    def __init__(self, inpDim):
        super(ClassificationModel, self).__init__()
        self.fc1 = nn.Linear(inpDim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 3)  # 3 output classes

    def forward(self, inp):
        x = torch.relu(self.fc1(inp))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    

class dataSetGen():
    def __init__(self, allInp, allOut):
        self.allInp = allInp
        self.allOut = allOut
    
    def __len__(self):
        return len(self.allInp)

    def __getitem__(self, idx):
        return {
            "inp": self.allInp[idx],
            "out": self.allOut[idx]
        }
    

def createDataSet(allDocs, allTargetChoice, allTargetSpans, hypothesis):
    allInp = []
    allOut = []

    # Creating inputs and outputs:
    for docID, doc in enumerate(allDocs):
        for segID, segment in enumerate(doc):
            for hypID, hypo in enumerate(hypothesis):
        # Do hyp first then seg so that all segs for a particular hyp are together (easy for task 1)
                input = "[CLS] " + hypo + " [SEP] " + segment + " [SEP]"
                input = getDocEmbeddings(input)
                if segID in allTargetSpans[docID][hypID]:
                    output = allTargetChoice[docID][hypID]
                else:
                    output = targetChoices["NotMentioned"]
                allInp.append(input)
                allOut.append(output)
    return allInp, allOut                


def train(allDocs, allTargetChoice, allTargetSpans, hypothesis, segIndices, mainModel):

    # I need to get the hypothesis and do:
    # [CLS] hypothesis [SEP] segment [SEP]
    # Then pass to model

    MainModel = SentenceTransformer('all-MiniLM-L6-v2')

    allInp, allOut = createDataSet(allDocs, allTargetChoice, allTargetSpans, hypothesis)
    trainDataSet = dataSetGen(allInp, allOut)
    trainLoader = DataLoader(trainDataSet, batch_size=BatchSize, shuffle=True)

    # As SBert
    inpDim = 384
    classiModel = ClassificationModel(inpDim)
    lossFunc = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(classiModel.parameters(), lr=0.001)

    for epoch in range(10):
        classiModel.train()
        totalLoss = 0

        for batch in trainLoader:
            inp = batch["inp"]
            out = batch["out"]

            optimizer.zero_grad()
            predicted = classiModel(inp)
            loss = lossFunc(predicted, out)
            loss.backward()
            optimizer.step()

            totalLoss += loss.item()

        print("Epoch: "+str(epoch)+" Loss: "+str(totalLoss))


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



if __name__ == '__main__':
    filePath = "./data/dev.json"
    hypoLabels = ["nda-11", "nda-16", "nda-15", "nda-10", "nda-2", "nda-1", "nda-19", "nda-12", "nda-20", "nda-3", "nda-18", "nda-7", "nda-17", "nda-8", "nda-13", "nda-5", "nda-4"]

    allDocs, allTargetChoice, allTargetSpans, hypothesis = loadDocs(filePath, hypoLabels)

    sBertModel = SentenceTransformer('all-MiniLM-L6-v2')

    segIndices = []
    # Getting index of segments
    for doc in allDocs:
        embeddings = getDocEmbeddings(doc)
        segIndices.append(segmentParas(embeddings, doc))

    # Now spans are no more the original spans. They are the new segments. Both in the document and in the target
    allDocs, allTargetSpans = adjustBySegment(allDocs, segIndices, allTargetSpans)

    train(allDocs, allTargetChoice, allTargetSpans, hypothesis, segIndices, sBertModel)