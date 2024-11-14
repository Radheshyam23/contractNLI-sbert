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

def relevanceLabelling(allDocs, allTargetChoice, allTargetSpans, hypothesis):
    allInp = []
    allOut = []

    # Not Relevant = 0
    # Relevant = 1

    # Creating inputs and outputs:
    for docID, doc in enumerate(allDocs):
        for hypID, hypo in enumerate(hypothesis):
            for segID, segment in enumerate(doc):
                input = hypo + " " + segment

                if segID in allTargetSpans[docID][hypID]:
                    output = 1
                else:
                    output = 0
                    
                allInp.append(input)
                allOut.append(output)
    return allInp, allOut


# Create dataset for hypothesis classification
def prepHypoClassiData(allDocs, allTargetChoice, allTargetSpans, hypothesis):
    """
    This is for train. Hence we will use the data directly from the dataset.
    
    The input data will be of form:
    ["hypo1 rel1 rel2 rel3 ...", "hypo2 rel1 rel2 rel3 ...", ...]
    so len(inp) will be numDocs * numHypothesis

    The target will be of form:
    [0, 1, 2, 0, 1, 2, ...]
    Just saying if each hypo is contra entail or not mentioned
    """

    allInp = []
    allOut = []

    # Not Relevant = 0
    # Relevant = 1

    # Creating inputs and outputs:
    for docID, doc in enumerate(allDocs):
        for hypID, hypo in enumerate(hypothesis):
            inp = hypo
            for segID in allTargetSpans[docID][hypID]:
                inp += " " + doc[segID]
            allInp.append(inp)
            allOut.append(allTargetChoice[docID][hypID])
    return allInp, allOut

def testHypoClassiData():
    """
        For inference. After Task 1 we will get a list of all relevant segments IDs.
        We will have to find only those segment IDs and then create the input.
    """
    pass