import spacy
from spacy import Language
from spacy.tokens import Doc
from spacy.tokenizer import Tokenizer

from utils import *


# dependencies that we take without further conditions, except for dep_cond
dep_signal = ['aux']
dep_subj = ['nsubj', 'csubj', 'nsubjpass', 'csubjpass']
dep_obj = ['dobj', 'pobj', 'iobj', 'dative', 'agent', 'ccomp', 'xcomp', 'attr']
dep_verb = ['neg', 'auxpass', 'prt', 'cc', 'conj', 'acomp']
dep_cond = ['acl', 'relcl', 'advcl', 'nmod', 'npadvmod', 'nounmod', 'advmod', 'prep'] # these could also be non-conditions, we do additional triggers


def find_verb_deco(root):
    # find verb (root) and its decorations, e.g. be, not, etc.
    direct_decos = tokens_with_dep(root.children, dep_verb) + [root]

    # decide whether the adverbial modifier is a verb or a condition
    for token in tokens_with_dep(root.children, ['advmod']):
        if not Phrase(token).has_trigger(condition_trigger):
            direct_decos.append(token)
            
    direct_decos.sort(key=by_index)

    # recursively find all of my decorations's own decorations (e.g. know not very well)
    all_decos = []
    for deco in direct_decos:
        if deco == root:
            all_decos.append(root)
        else:
            all_decos += find_verb_deco(deco)
    
    return all_decos


def find_times_conds(potential_cond):
    times = []
    conds = []
    for cond_phrase in potential_cond:
        if len(cond_phrase) > 1:
            if cond_phrase.has_trigger(condition_trigger) and cond_phrase.as_str().strip() not in condition_trigger:
                if cond_phrase.has_trigger(time_point):
                    times.append(cond_phrase)
                else:
                    conds.append(cond_phrase)
    return times, conds


def extract(root, extract_obj_recursively=True):
    '''
    extract the whole phrase, subject, signal word, verb, time, condition, and object
    if extract_obj_recursively is True, then objects is a list of Extracted
    if extract_obj_recursively is False, then objects is a list of Phrase
    '''

    # find verb phrase
    verb_phrase = Phrase(find_verb_deco(root), has_skips=True)
    
    # find signal word
    signal_word = Phrase(tokens_with_dep(root.children, dep_signal))
    if signal_word.as_str().strip() not in signal: # not in the whitelist, add to verb instead
        verb_phrase += signal_word
        verb_phrase.tokens.sort(key=by_index)
        signal_word = Phrase([])

    # subject, object and conditions should be a child of the verb phrase
    verb_phrase_children = verb_phrase.get_children()

    # find subject phrases
    subj_phrases = phrases_from_roots(tokens_with_dep(verb_phrase_children, dep_subj))


    # find object phrases
    obj_phrases = phrases_from_roots(tokens_with_dep(verb_phrase_children, dep_obj))

    # prep could either be object or condition/time
    for phrase in phrases_from_roots(tokens_with_dep(verb_phrase_children, ['prep'])):
        if not phrase.starts_with_trigger(condition_trigger):
            obj_phrases.append(phrase)


    # find conditions and time constraints
    potential_cond_roots = set(tokens_with_dep(verb_phrase_children, dep_cond)) - set(obj.root for obj in obj_phrases)
    times, conds = find_times_conds(phrases_from_roots(potential_cond_roots))


    # for each objects (given their roots), extract their subj, signal, verb, conds, times, and obj
    objects = []

    if extract_obj_recursively:
        for obj_phrase in obj_phrases:
            # clausal compliments we can directly extract as sentence
            if obj_phrase.root.dep_ in ['ccomp', 'xcomp']:
                extracted_obj = extract(obj_phrase.root, extract_obj_recursively=False)

                # if the extracted part has no new subject, the verbs are mergeable and there's only 1 object, merge the verbs with the main and re-extract
                extracted_signal_verb = extracted_obj.signal_word + extracted_obj.verb_phrase
                if extracted_obj.subject_phrases == [] \
                    and verb_phrase[-1].nbor(1) == extracted_signal_verb[0] \
                    and len(extracted_obj.objects) == 1:
                    verb_phrase += extracted_signal_verb
                    times += extracted_obj.times
                    conds += extracted_obj.conditions

                    # extract one layer deeper to get the object's object
                    extracted_obj = extract(extracted_obj.objects[0].root, extract_obj_recursively=False)

                objects.append(extracted_obj)

            # non-clausal objects might still contain relative clauses (or acl) or conditions
            else:
                # look for a relative clause and only take the highest level clause, in case it is nested (lowest depth)
                rel_clause = min(tokens_with_dep(obj_phrase, ['relcl', 'acl']), default=None, key=by_depth)

                if rel_clause is not None:
                    rel_clause = Phrase(rel_clause)

                    # extract only the relative clause (not the whole object phrase!)
                    extracted_obj = extract(rel_clause.root, extract_obj_recursively=False)

                    # analyze the rest of the object phrase
                    rest_of_obj_phrase = obj_phrase - rel_clause # obj phrase WITHOUT the rel clause
                    if obj_phrase.root.dep_ == 'prep': # for prepositional object phrases, remove the preposition (e.g. for)
                        rest_of_obj_phrase -= obj_phrase.root

                    # find any potential conditions in the rest of the object phrase
                    t, c = find_times_conds(phrases_from_roots(tokens_with_dep(rest_of_obj_phrase, dep_cond)))
                    t = [phrase for phrase in t if phrase.starts_with_trigger(condition_trigger)]
                    c = [phrase for phrase in c if phrase.starts_with_trigger(condition_trigger)]
                    extracted_obj.times += t
                    extracted_obj.conditions += c

                    # find the referred noun phrase
                    ref_noun_phrase = rest_of_obj_phrase - (Phrase.merge(t) + Phrase.merge(c))

                    # find the relative word (which, who, etc.) and prepend it with the referred noun phrase
                    for phrases in [extracted_obj.subject_phrases, extracted_obj.objects]:
                        for i, phrase in enumerate(phrases):
                            # find the position of the first relative word in the phrase (if any)
                            rel_index = next((i for i, t in enumerate(phrase) if Phrase(t).has_trigger(relative_words)), None)
                            if rel_index is not None:
                                phrases[i] = Phrase(phrase[:rel_index]) + ref_noun_phrase + Phrase(phrase[rel_index:])

                    # only the relative clause was extracted, but the whole phrase should still be the whole object phrase
                    extracted_obj.whole_phrase = obj_phrase

                    objects.append(extracted_obj)

                else:
                    # no clauses found to extract, but still look for conditions and times
                    t, c = find_times_conds(phrases_from_roots(tokens_with_dep(obj_phrase, dep_cond)))
                    t = [phrase for phrase in t if phrase.starts_with_trigger(condition_trigger)]
                    c = [phrase for phrase in c if phrase.starts_with_trigger(condition_trigger)]
                    objects.append(Extracted(obj_phrase, None, None, None, t, c, None))

    else: # do not analyse my objects, instead just return the object phrases
        objects = obj_phrases

    
    return Extracted(Phrase(root), subj_phrases, signal_word, verb_phrase, times, conds, objects)


def extract_sentence(doc):
    root = find_root(doc)
    assign_depth(root)

    return extract(root)


# register custom extension attributes
Doc.set_extension('extracted', default=None)
Doc.set_extension('replacements', default=None)


@Language.component('phrase_spans')
def phrase_spans(doc):
    root = find_root(doc)
    assign_depth(root)

    extracted = extract(root)
    spans = reduce(lambda x,y: x+y, extracted.as_span(), []) # all spans in one list
    doc.spans['sc'] = spans # create a new span group in the doc
    doc._.extracted = extracted # store the extracted sentence in the doc

    return doc

# for Prodigy to find the span group name
phrase_spans.key = 'sc'

# experimental pre- and post-processing pipeline components, use with_preprocessing instead
# @Language.component('preprocess', retokenizes=True)
# def preprocess(doc):
#     sentence, replacements = simplify_sentence(doc.text)
#     tokenizer = Tokenizer(doc.vocab)
#     doc = tokenizer(sentence)
#     doc._.replacements = replacements
#     return doc

# @Language.component('postprocess', retokenizes=True)
# def postprocess(doc):
#     if doc._.replacements is not None:
#         doc = restore_sentence(doc, doc._.replacements)
#     return doc