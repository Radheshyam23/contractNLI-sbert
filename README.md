# contractNLI-sbert
Contract NLI using sentence transformers instead of using spans and sliding context.

The Plan:
- First break down the doc into paragraphs
- Now use sbert to encode each paragraph
- compare the encodings for similarity
- decide a threshold similarity
- club similar paras into one segment.

Model training:
- Model such as AlBERT, RoBERTA, etc.
- Figure out how to aggregate



Observations:
- Spans doesn't exactly correlate with new line. There are some spans which are abruptyly in the middle of a sentence.