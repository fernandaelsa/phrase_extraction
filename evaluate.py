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
add_span_label_vocabs(nlp)


examples = []

# load and iterate over jsonl file
with open('dataset/annotated_gold_standard.jsonl', 'r') as f:
    for line in f:
        d = json.loads(line)
        sentence = d['text']
        # doc_pred = nlp(sentence)
        doc_pred = with_preprocessing(nlp, sentence)
        doc_gold = doc_from_annotation(nlp.vocab, d)
        
        examples.append(Example(predicted=doc_pred, reference=doc_gold))

scores = Scorer.score_spans(examples, 'sc', getter=lambda doc, attr: doc.spans['sc'], allow_overlap=True)

# print scores rounded to 3 decimal places
print(f"Precision: {scores['sc_p']:.3f}")
print(f"Recall: {scores['sc_r']:.3f}")
print(f"F1: {scores['sc_f']:.3f}")

print("By category:")
for label in categories:
    print(f"{label}:")
    for metric in scores['sc_per_type'][label]:
        print(f"    {metric}: {scores['sc_per_type'][label][metric]:.3f}")