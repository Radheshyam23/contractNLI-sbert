# Load and parse the json file:
import json
import os
import torch
from sklearn.metrics.pairwise import cosine_similarity
import torch.nn as nn
from torch.utils.data import DataLoader
from datasett import DataSetGen, loadDocs
from segmentation import getDocEmbeddings, segmentParas, adjustBySegment
from hardcode import targetChoices, hypoLabels

from sentence_transformers import SentenceTransformer


################

BatchSize = 16

################


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


if __name__ == '__main__':
    filePath = "./data/dev.json"

    allDocs, allTargetChoice, allTargetSpans, hypothesis = loadDocs(filePath, hypoLabels)

    # If You do NOT want to perform segmentation, comment the following lines
    #########################
    # segmenter = segmentation(all)    
    sBertModel = SentenceTransformer('all-MiniLM-L6-v2')
    segmentThreshold = 0.6
    segIndices = []
    # Getting index of segments
    for doc in allDocs:
        embeddings = getDocEmbeddings(doc)
        segIndices.append(segmentParas(embeddings, segmentThreshold))

    # Now spans are no more the original spans. They are the new segments. Both in the document and in the target
    allDocs, allTargetSpans = adjustBySegment(allDocs, segIndices, allTargetSpans)
    #########################

    

    # train(allDocs, allTargetChoice, allTargetSpans, hypothesis, segIndices, sBertModel)