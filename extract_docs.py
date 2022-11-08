import spacy
import sys
import os
import re
from string import punctuation, whitespace
import pandas as pd
from spacy import displacy
from phrase_extraction import *

# annotation labels of spans
categories = ['SENTENCE', 'SUBJECT', 'SIGNAL', 'VERB', 'TIME', 'CONDITION', 'OBJECT', 'OP_SUBJECT', 'OP_SIGNAL', 'OP_VERB', 'OP_TIME', 'OP_CONDITION', 'OP_OBJECT']


def read_documents(directory): 
  '''reads in txts of regulatory and realization documents
  Input: multiple .txt files (each a sentence)
  Output: dictionary with file name as key and its content as value'''
  doc_dict = dict()
  files = os.listdir(directory)
  try:
    for fi in files:
        if fi.endswith('.txt'):
          with open(directory+'/'+fi,'r') as f:
              doc_dict[re.sub('\.txt', '', fi)] = f.read()
  except FileNotFoundError:
    print("Wrong file or file path to dir.")
    quit()
  return doc_dict


def extract_document(folder_name):
  # read all .txt files from the input folder
  input_path = './input/' + folder_name
  documents = read_documents(input_path)

  for file in documents:
    sentence = documents[file]
    doc = nlp(sentence)
    all_spans = doc.spans['sc']

    spans_dict = {'SENTENCE': sentence}
    for label, spans in groupby(all_spans, lambda span: span.label_):
        # merge all spans of one category into one string
        merged = ' | '.join([span.text.strip(punctuation + whitespace) for span in spans])
        spans_dict[label] = merged
    
    # table by category
    table_html = html_table(pd.DataFrame(
          {
              "EXTRACTED": [spans_dict.get(category, '') for category in categories],
          },
          index = categories
      ))
    
    # as highlighted spans
    span_html = displacy.render(doc, style='span', options={'colors': span_colors})
    
    # write into result folder
    out_path = f'./result/{folder_name}/{file}.html'
    os.makedirs(os.path.dirname(out_path), exist_ok=True) # create folder if not exist
    with open(out_path, 'w') as f:
      f.write(f'{span_html}\n\n<br/><hr><br/>\n\n{table_html}')


if __name__ == '__main__':
  # load the previously saved model:
  if len(sys.argv) != 2:
    print("Wrong number of arguments. Please specify the path to spacy model that will extract the spans.")
    quit()
  nlp = spacy.load(sys.argv[1])

  # extract each document from the input folder and save it as html in result folder
  extract_document('realization_document')
  extract_document('regulatory_document')