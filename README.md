# Phrase Extraction

## Overview

### Purpose
This repository contains the implementation for the Bachelor's Thesis *Phrase Extraction for In-Depth Comparison of Regulatory Documents and Their Realizations*.

The aim of this project is to help automate legal compliancy check between regulatory text and their realizations, such as for GDPR and Privacy Policy texts. More specifically, it aims to automatically extract syntactically and semantically relevant phrases from sentences, from which phrase-to-phrase comparison between sentences could be done.

### The 3 Methods
We present three different methods, Method A, Method B and Method C:  
Method A is a hand-engineered algorithm based on dependency tree parsing, and is the most accurate and robust out of the three.  
Method B is a simple machine learning method that is the most efficient to run, but has low accuracy.  
Method C is also a machine learning method that combines the ideas of the former two approaches (dependency tree parsing and trainable span categorizer) to find a balanced trade-off between accuracy and complexity. It achieves decent results while not being as hard to understand as Method A.

### What's included?
- `phrase_extraction.py`: Implementation of Method A. You must import this script in order to use Method A.
- `span_suggester.py`: Implementation of the span suggester for Method C. The implementation is from [spaCy's experimental repository](https://github.com/explosion/spacy-experimental). You must import this script in order to use Method C.
- `create_models.ipynb`: A convenient Jupyter Notebook you can use to create the spaCy models for each methods. It contains detailed instructions how to save and load the model for Method A, and how to train a model for Method B or Method C. It also contains instructions on how to use Prodigy to annotate training dataset.
- `visualize.ipynb`: Another Jupyter Notebook that you can use to test the models and see the extracted results visualized as spans.
- `extract_docs.py`: Run this script in the command line with the single argument as the path to the model to use to extract the sentences. See below for more detailed explanation.
- `evaluate.py`: Run this script in the command line with the single argument as the path to the model to evaluate. The model will be evaluated against the gold standard (`./dataset/annotated_gold_standard.jsonl`).
- `utils.py`: Contains many helper classes, functions and data used by Method A.

## Setup

This project uses Python 3.9 and [spaCy](https://spacy.io/usage) v3.4, but it should be compatible with similar versions as well. It is recommended to use a virtual environment such as Anaconda. To download and install spaCy and the required pretrained transformer model, run the following:

```
pip install -U pip setuptools wheel
pip install -U spacy
python -m spacy download en_core_web_trf
```

Optionally, if you plan on annotating data, install [Prodigy](https://prodi.gy/) - Academic License can be requested by emailing contact@explosion.ai.


## Creating the Models for Method A, B, C

Although the pre-annotated dataset is already included in this repository, you must create the models yourself, due to their large file sizes. See the `create_models.ipynb` Jupyter Notebook to see how. You can simply run the code snippets/commands in the notebook to export/train the models.

Note that the trained methods (B and C) each contain 2 models by default: `model-best` and `model-last`. You must specify which directory when loading the model in spaCy (e.g. `spacy.load('models/method_b/model-best')`). It is recommended to use `model-best`, since it should achieve the highest average score.

If you do not want to train the models for Methods B and C, then you could **download our pretrained model** from [here](https://syncandshare.lrz.de/getlink/fi8nZtRYtLaDpuctW6hMa4/method_bc.zip) and extract them into the `models/` directory.


## Extracting from input text files and saving as .html

Put each sentence into a .txt file (one sentence per file) in the `input/realization_document/` and `input/regulatory_document/` directories.

Then, to extract all input text files using the model, run the Python script with the path to the above saved model:
```
python extract_docs.py <model_path>
```

You can find the resulting .html files in the `result/` directory. the `result/index.html` file contains all links to the results and their preview embedding, so you can view all results at once. Simply open this in the browser.


## Evaluating a model (Precision/Recall/F-Score)

Once you have a model, to evaluate it against the Gold Standard, call the following script with the model path:
```
python evaluate.py <model_path>
```
This will print the F1-Score to the console.