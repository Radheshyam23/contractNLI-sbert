# Load and parse the json file:
import json
import os
import torch
from sklearn.metrics.pairwise import cosine_similarity

from sentence_transformers import SentenceTransformer

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

        
    segEmbeds = torch.tensor([])
    for i in range(len(segments)-1):
        if (segments[i+1] - segments[i] == 1):
            segEmbeds = torch.cat((segEmbeds, embeddings[segments[i]].unsqueeze(0)), 0)
        else:
            print(segments[i], segments[i+1])
            currSeg = " ".join(doc[segments[i]:segments[i+1]])
            newEmbed = getDocEmbeddings(currSeg)
            print(newEmbed.shape)
            segEmbeds = torch.cat((segEmbeds, newEmbed.unsqueeze(0)), 0)
    
    # Last Segment
    if (segments[-1] == numSent-1):
        segEmbeds = torch.cat((segEmbeds, embeddings[segments[-1]].unsqueeze(0)), 0)
    else:
        print(segments[i], segments[i+1])
        currSeg = " ".join(doc[segments[-1]:])
        newEmbed = getDocEmbeddings(currSeg)
        segEmbeds = torch.cat((segEmbeds, newEmbed.unsqueeze(0)), 0)

        
    return segEmbeds
    

if __name__ == '__main__':
    filePath = "./data/dev.json"
    allDocs = loadDocs(filePath)

    sBertModel = SentenceTransformer('all-MiniLM-L6-v2')  # You can choose other pre-trained models as well  

    allEmbeds = []

    for doc in allDocs:
        embeddings = getDocEmbeddings(doc)
        embeddings = segmentParas(embeddings, doc)
        allEmbeds.append(embeddings)

    print(len(allEmbeds))