import sys
import spacy
from spacy.tokens import Doc, Span
from spacy.training import Example
from spacy.scorer import Scorer
from phrase_extraction import *
import json

# load model to evaluate
if len(sys.argv) != 2:
    print("Wrong number of arguments. Please specify the path to spacy model that will extract the spans.")
    quit()
nlp = spacy.load(sys.argv[1])


examples = []

# load and iterate over jsonl file
with open('dataset/annotated_gold_standard.jsonl', 'r') as f:
    for line in f:
        d = json.loads(line)
        sentence = d['text']
        doc_pred = nlp(sentence)
        doc_gold = nlp(sentence, disable=['phrase_spans', 'spancat']) # we just want the tokenization TODO: can just use another simple nlp and then pass Alignment.from_strings in Example constructor below
        spans = []
        for s in d['spans']:
            spans.append(Span(doc_gold, s['token_start'], s['token_end']+1, label=s['label']))
        doc_gold.spans['sc'] = spans
        examples.append(Example(predicted=doc_pred, reference=doc_gold))

scores = Scorer.score_spans(examples, 'sc', getter=lambda doc, attr: doc.spans['sc'], allow_overlap=True)

print(f"Precision: {scores['sc_p']}")
print(f"Recall: {scores['sc_r']}")
print(f"F1: {scores['sc_f']}")
for label in scores['sc_per_type']:
    print(f"{label}: {scores['sc_per_type'][label]}")