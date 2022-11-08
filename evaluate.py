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

# same model used to create the gold standard, so it will tokenize the same way
nlp_gold = spacy.load('en_core_web_trf')
nlp_gold.add_pipe('merge_noun_chunks')
nlp_gold.add_pipe('merge_entities')
add_span_label_vocabs(nlp_gold)


examples = []

# load and iterate over jsonl file
with open('dataset/annotated_gold_standard.jsonl', 'r') as f:
    for line in f:
        d = json.loads(line)
        sentence = d['text']
        doc_pred = nlp(sentence)
        doc_gold = nlp_gold(sentence) # we just want the tokenization
        
        spans = []
        for s in d['spans']:
            spans.append(Span(doc_gold, s['token_start'], s['token_end']+1, label=s['label']))
        doc_gold.spans['sc'] = spans
        examples.append(Example(predicted=doc_pred, reference=doc_gold))

        for span in doc_gold.spans['sc']:
            if span.label == 6110311367300086286:
                print(span.text)

scores = Scorer.score_spans(examples, 'sc', getter=lambda doc, attr: doc.spans['sc'], allow_overlap=True)

print(f"Precision: {scores['sc_p']}")
print(f"Recall: {scores['sc_r']}")
print(f"F1: {scores['sc_f']}")
for label in scores['sc_per_type']:
    print(f"{label}: {scores['sc_per_type'][label]}")