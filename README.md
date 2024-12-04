# contractNLI-sbert
An alternative approach to the problem proposed in the [ContractNLI paper](https://aclanthology.org/2021.findings-emnlp.164/).

We use the [ContractNLI dataset](https://stanfordnlp.github.io/contract-nli/)

#### The task:
Given a legal document and a hypothesis, we need to find out if that hypothesis contradicts the document or is entailed by it or not-mentioned.

#### The dataset:
Has 607 annotated non-disclosure agreements (NDAs). 17 hypothesis to test against.
The document is further split into "spans" which is roughly based on the newline characters (so each paragraph is a span).

### Our approach:
Splitting the task into two parts:

1. Relevance Classification:
Given a hypothesis and each span in the document, classify if that span is relevant to that hypothesis or not.

2. Hypothesis Classification:
Given a hypothesis and the relevant spans, classify the hypothesis as contradiction, entailment or not mentioned.

Entire pipeline:
Given a document and the hypothesis, first perform relevance classification to identify the relevant spans. Now take the relevant spans and the hypothesis and perform Hypothesis Classification.

### Other Experiments

Segmentation:
Re-defining spans to be more meaningful and encapsulate a larger context based on meaning. We calculate the similarity between consecutive spans by using sentence-transformer and if they are similar enough, club it into a bigger span (which we are calling as a segment.)