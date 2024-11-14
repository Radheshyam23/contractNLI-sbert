from torch.utils.data import Dataset
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

##############
max_length = 145
# The hypo + relevant spans on an average have length of 60 and 90 percentile are at 137.
##############


class HypothesisClassificationDataset(Dataset):
    def __init__(self, inputs, labels, tokenizer, max_length):
        """
        Args:
        - inputs (list of str): List of "hypothesis relevantSentence1 relevantSentence2 ..." strings.
        - labels (list of int): List of target labels (0, 1, 2) for each input.
        - tokenizer: Hugging Face tokenizer.
        - max_length (int): Max token length for the input.
        """
        self.inputs = inputs
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        input_text = self.inputs[idx]
        label = self.labels[idx]

        # Tokenize the input
        encoded = self.tokenizer(
            input_text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        item = {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long),
        }
        return item



# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
hypothesis_model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=3)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./hypothesis_results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    num_train_epochs=10,
    weight_decay=0.01,
    logging_dir="./hypothesis_logs",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
)

# Metrics for evaluation
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="weighted")
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}

# Function to train the model
def train_hypothesis_classification(train_inputs, train_labels, dev_inputs, dev_labels):

    # Prepare datasets
    train_dataset = HypothesisClassificationDataset(train_inputs, train_labels, tokenizer, max_length)
    dev_dataset = HypothesisClassificationDataset(dev_inputs, dev_labels, tokenizer, max_length)

    # Initialize the Trainer
    trainer = Trainer(
        model=hypothesis_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        compute_metrics=compute_metrics,
    )

    # Train and save the model
    trainer.train()
    trainer.save_model("./hypothesis_classifier")

def classify_hypothesis(input_text, tokenizer, model):
    """
    Classifies a single hypothesis with relevant sentences into one of 3 classes.

    Args:
    - input_text (str): Input string of the form "hypothesis relevantSentence1 relevantSentence2 ...".
    - tokenizer: Hugging Face tokenizer.
    - model: Trained hypothesis classification model.
    - max_length (int): Max token length.

    Returns:
    - Predicted label (0, 1, or 2).
    """
    model.eval()

    # Tokenize the input
    encoded = tokenizer(
        input_text,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )

    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]

    # Perform inference
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        pred_label = outputs.logits.argmax(dim=-1).item()

    return pred_label