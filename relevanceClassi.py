from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, EarlyStoppingCallback
from torch.utils.data import DataLoader, Dataset
import torch
from transformers import Trainer, TrainingArguments
from torch.utils.data import random_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

##########
max_length = 75
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
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",  # Save model at each epoch
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    num_train_epochs=10,  # Maximum number of epochs
    weight_decay=0.01,
    logging_dir="./logs",
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
    preds = pred.predictions.argmax(-1)  # Predicted labels
    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}

def trainRelevance(trainInputs, trainLabels, devInputs, devLabels):
    trainRelDataset = RelevanceDataset(trainInputs, trainLabels, tokenizer, max_length=max_length)
    devRelDataset = RelevanceDataset(devInputs, devLabels, tokenizer, max_length=max_length)

    example = trainRelDataset[0]
    print("Input IDs:", example["input_ids"])
    print("Attention Mask:", example["attention_mask"])
    print("Label:", example["labels"])

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


def getRelevanceLabels(testInputs, relClassiModelPath):
    """
    Predicts relevance labels (1 or 0) for a list of hypothesis + sentence pairs.
    """

    tokenizer = AutoTokenizer.from_pretrained(relClassiModelPath)
    relevance_model = AutoModelForSequenceClassification.from_pretrained(relClassiModelPath)

    # Ensure the model is in evaluation mode
    relevance_model.eval()

    # Tokenize the inputs
    encoded_inputs = tokenizer(
        testInputs,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )

    # Move tensors to the device used by the model (e.g., GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    relevance_model.to(device)
    input_ids = encoded_inputs["input_ids"].to(device)
    attention_mask = encoded_inputs["attention_mask"].to(device)

    # Perform inference
    with torch.no_grad():
        outputs = relevance_model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits

    # Get predicted labels (1 = Relevant, 0 = Not Relevant)
    predicted_labels = torch.argmax(logits, dim=-1).cpu().tolist()

    return predicted_labels