import spacy
import sys
import os
import re
from string import punctuation, whitespace
import pandas as pd
from spacy import displacy
from phrase_extraction import *

# annotation labels of spans
categories = ['SUBJECT', 'SIGNAL', 'VERB', 'TIME', 'CONDITION', 'OBJECT', 'OP_SUBJECT', 'OP_SIGNAL', 'OP_VERB', 'OP_TIME', 'OP_CONDITION', 'OP_OBJECT']


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
  input_path = 'input/' + folder_name
  documents = read_documents(input_path)

  result_file_paths = []
  # extract each sentence and save in results folder
  for file in documents:
    sentence = documents[file]
    doc = nlp(sentence)
    all_spans = doc.spans['sc']

    spans_dict = dict()
    for label, spans in groupby(all_spans, lambda span: span.label_):
        # merge all spans of one category into one string
        merged = ' | '.join([span.text.strip(punctuation + whitespace) for span in spans])
        spans_dict[label] = merged
    
    # table by category
    table_html = html_table(pd.DataFrame(
          {
              "EXTRACTED": [sentence] + [spans_dict.get(category, '') for category in categories],
          },
          index = ['SENTENCE'] + categories
      ))
    
    # as highlighted spans
    span_html = displacy.render(doc, style='span', options={'colors': span_colors})
    
    # write into result folder
    out_path = f'{folder_name}/{file}.html'
    result_file_paths.append(out_path)
    out_path = 'result/' + out_path
    os.makedirs(os.path.dirname(out_path), exist_ok=True) # create folder if not exist
    with open(out_path, 'w') as f:
      f.write(f'{span_html}\n\n<br/><hr><br/>\n\n{table_html}')
    
  return result_file_paths


if __name__ == '__main__':
  # load the previously saved model:
  if len(sys.argv) != 2:
    print("Wrong number of arguments. Please specify the path to spacy model that will extract the spans.")
    quit()
  nlp = spacy.load(sys.argv[1])

  # extract each document from the input folder and save it as html in result folder
  result_paths = []
  result_paths += extract_document('realization_document')
  result_paths += extract_document('regulatory_document')

  # create a convinient index.html file to navigate through the results
  with open('result/index.html', 'w') as f:
    index = '<html><body><h1>Results</h1>'
    for path in result_paths:
      index += f'<a href="{path}">{path}</a><br/>'
    index += '</body></html>'
    f.write(index)