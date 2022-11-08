# phrase_extraction

## Extracting from text files

Put each sentence into a .txt file (one sentence per file) in the `input/realization_document/` and `input/regulatory_document/` folders.

Additionally, save the spaCy model with the span categorization pipeline under the `models/` folder.

Finally, to extract all input text files using the model, run the Python script with the path to the above saved model:
```
python extract_docs.py <model_path>
```

You can find the resulting .html files in the `result/` folder.


## Evaluating a model (Precision/Recall/F-Score)

Once you have a model, to evaluate it against the Gold Standard, call the following script with the model path:
```
python evaluate.py <model_path>
```


## Annotating using Prodigy

Using the suggestions of the Method A that you can manually correct:
```
prodigy spans.correct initial_gold ./phrase_spans_model ./dataset/gold_standard.jsonl --label SUBJECT,SIGNAL,VERB,TIME,CONDITION,OBJECT,OP_SUBJECT,OP_SIGNAL,OP_VERB,OP_TIME,OP_CONDITION,OP_OBJECT -F phrase_extraction.py -c phrase_spans
```

Exporting annotated data to json
```
prodigy db-out news_headlines > ./news_headlines.jsonl
```

Train directly with prodigy:
```
prodigy train ./output_model --spancat initial_gold
```

Exporting the annotation database as spacy file:
```
prodigy data-to-spacy ./output_dataset --spancat initial_gold
```