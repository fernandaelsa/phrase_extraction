# Phrase Extraction

## (TODO) spaCy Setup

```
pip install -U pip setuptools wheel
pip install -U spacy
python -m spacy download en_core_web_trf
```

Optionally, install [Prodigy](https://prodi.gy/) - Academic License can be requested by emailing contact@explosion.ai.

## Creating the models for method A, B, C

Although the pre-annotated dataset is already included in this repository, you must create the models yourself, due to their large file sizes. See the `create_models.ipynb` Jupyter Notebook to see how. You can simply run the code snippets/commands in the notebook to export/train the models.


## Extracting from input text files and saving as .html

Put each sentence into a .txt file (one sentence per file) in the `input/realization_document/` and `input/regulatory_document/` folders.

Then, to extract all input text files using the model, run the Python script with the path to the above saved model:
```
python extract_docs.py <model_path>
```

You can find the resulting .html files in the `result/` folder.


## Evaluating a model (Precision/Recall/F-Score)

Once you have a model, to evaluate it against the Gold Standard, call the following script with the model path:
```
python evaluate.py <model_path>
```