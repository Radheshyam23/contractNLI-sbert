# contractNLI-sbert
Contract NLI using sentence transformers instead of using spans and sliding context.

1. Relevance Classification:
Given a hypothesis and each span in the document, classify if that span is relevant to that hypothesis or not.

2. Hypothesis Classification:
Given a hypothesis and the relevant spans, classify the hypothesis as contradiction, entailment or not mentioned.

3. Segmentation:
Re-defining spans to be more meaningful and encapsulate a larger context based on meaning. We calculate the similarity between
consecutive spans and if they are similar enough, club it into a bigger span (which we are calling as a segment.)
This is an add on. Can be enabled by uncommenting the relevant lines of code.