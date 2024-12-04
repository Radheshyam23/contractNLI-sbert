from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, EarlyStoppingCallback, TrainerCallback
from torch.utils.data import DataLoader, Dataset
import torch
from transformers import Trainer, TrainingArguments
from torch.utils.data import random_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tqdm import tqdm
import numpy as np
from sklearn.metrics import precision_recall_curve, average_precision_score

##########
# For non-Segmentation 75
max_length = 75
# For Segmentation 105
# max_length = 105
##########


# Tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

class RelevanceDataset(Dataset):
    def __init__(self, inputs, labels, tokenizer, max_length):
        self.inputs = inputs
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        # Tokenize hypothesis + sentence pair
        encoded = self.tokenizer(
            self.inputs[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        # Return input IDs, attention mask, and label
        item = {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }
        return item



# Relevance classification model
relevanceModel = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

training_args = TrainingArguments(
    output_dir="./relev_results",
    evaluation_strategy="epoch",
    save_strategy="epoch",  # Save model at each epoch
    save_total_limit=1,  # Only keep one model checkpoint
    learning_rate=2e-5,
    per_device_train_batch_size=50,
    num_train_epochs=10,  # Maximum number of epochs
    weight_decay=0.01,
    logging_dir="./relev_logs",
    load_best_model_at_end=True,  # Automatically load the best model
    metric_for_best_model="eval_loss",  # Monitor evaluation loss
    greater_is_better=False,  # Lower loss is better
)

# Early stopping callback
early_stopping_callback = EarlyStoppingCallback(
    early_stopping_patience=3  # Stop training if no improvement for 3 evaluation steps
)

def compute_metrics(pred):
    labels = pred.label_ids
    predictions = pred.predictions
    scores = predictions[:, 1]  # Probability of being relevant
    preds = pred.predictions.argmax(-1)  # Predicted labels

    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")
    
    # mAP and P@R80 calculations
    # For binary classification, treat the problem as "label 1 vs. label 0"
    ap = average_precision_score(labels, scores)  # mAP
    precision_vals, recall_vals, _ = precision_recall_curve(labels, scores)
    
    try:
        idx = np.where(recall_vals >= 0.8)[0][0]  # Index of first recall >= 0.8
        p_at_r80 = precision_vals[idx]
    except IndexError:
        p_at_r80 = 0.0  # If recall never reaches 0.8

    with open("relevance_metrics.txt", "a") as f:
        f.write(f"Accuracy: {acc}, Precision: {precision}, Recall: {recall}, F1: {f1}, mAP: {ap}, P@R80: {p_at_r80}\n")

    # Return results as a dictionary
    return {
        "accuracy": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "mAP": ap,
        "P@R80": p_at_r80
    }

def trainRelevance(trainInputs, trainLabels, devInputs, devLabels):
    trainRelDataset = RelevanceDataset(trainInputs, trainLabels, tokenizer, max_length=max_length)
    devRelDataset = RelevanceDataset(devInputs, devLabels, tokenizer, max_length=max_length)

    # example = trainRelDataset[0]
    # print("Input IDs:", example["input_ids"])
    # print("Attention Mask:", example["attention_mask"])
    # print("Label:", example["labels"])

    trainer = Trainer(
        model=relevanceModel,
        args=training_args,
        train_dataset=trainRelDataset,
        eval_dataset=devRelDataset,
        compute_metrics=compute_metrics,
        callbacks=[early_stopping_callback],
    )

    trainer.train()
    trainer.save_model("./relevance_classifier")


def getRelevanceLabels(testInputs, relClassiModelPath, testLabels):
    """
    Predicts relevance labels (1 or 0) for a list of hypothesis + sentence pairs.
    This is for inference
    """

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    relevance_model = AutoModelForSequenceClassification.from_pretrained(relClassiModelPath)

    # Ensure the model is in evaluation mode
    relevance_model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    relevance_model.to(device)

    predictedLabels = []
    scores = []

    batchSize = 100

    for i in tqdm(range(0, len(testInputs), batchSize)):
        batchInputs = testInputs[i:i + batchSize]

        # Tokenize the inputs
        encodedInputs = tokenizer(
            batchInputs,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )

        inputIds = encodedInputs["input_ids"].to(device)
        attentionMask = encodedInputs["attention_mask"].to(device)

        # Perform inference
        with torch.no_grad():
            outputs = relevance_model(input_ids=inputIds, attention_mask=attentionMask)
            logits = outputs.logits
            probab = torch.nn.functional.softmax(logits, dim=-1)

        # Get predicted labels (1 = Relevant, 0 = Not Relevant)
        predictedLabels.extend(torch.argmax(logits, dim=-1).cpu().tolist())
        scores.extend(probab[:, 1].cpu().tolist())
    

    # Calculate metrics
    acc = accuracy_score(testLabels, predictedLabels)
    precision, recall, f1, _ = precision_recall_fscore_support(testLabels, predictedLabels, average="binary")

    # mAP calculation
    ap = average_precision_score(testLabels, scores)  # mAP

    # P@R80 calculation
    precision_vals, recall_vals, _ = precision_recall_curve(testLabels, scores)
    try:
        idx = np.where(recall_vals >= 0.8)[0][0]  # Index of first recall >= 0.8
        p_at_r80 = precision_vals[idx]
    except IndexError:
        p_at_r80 = 0.0  # If recall never reaches 0.8

    # Print metrics
    print("Relevance Classification Metrics:")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1: {f1:.4f}")
    print(f"mAP: {ap:.4f}")
    print(f"P@R80: {p_at_r80:.4f}")

    # Save metrics to a file
    with open("test_relevance_metrics.txt", "w") as f:
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"F1: {f1:.4f}\n")
        f.write(f"mAP: {ap:.4f}\n")
        f.write(f"P@R80: {p_at_r80:.4f}\n")


    return predictedLabels