# phrase_extraction


## annotating using Prodigy

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