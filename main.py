# Load and parse the json file:
import json
import os

def loadDocs(filePath):
    with open(filePath, 'r') as f:
        data = json.load(f)

    print("Number of documents in the dataset:"+str(len(data["documents"])))

    allDocs = []

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
        
    

    return allDocs


if __name__ == '__main__':
    filePath = "./data/dev.json"
    allDocs = loadDocs(filePath)

