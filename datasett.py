import json
from hardcode import targetChoices, hypoLabels

class DataSetGen():
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