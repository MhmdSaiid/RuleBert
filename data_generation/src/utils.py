from typing import Any, Dict, Tuple, Union, List
from clyngor import ASP
from random import shuffle
from .Triple import Triple
from .Rule import Rule
import numpy as np
import re
import subprocess
import pickle
import os
import datetime
from scipy.special import softmax
import hashlib

# put text of the rule here

# TODO Add DocString and Type Hinting

with open('data_generation/data/rule2text.pkl', 'rb') as f:
    rule2text = pickle.load(f)

# set of symmetric predicates
with open('data_generation/data/symmetric_preds_set.pkl', 'rb') as f:
    symmetric_preds_set = pickle.load(f)

# set of nonsymmetric predicates
with open('data_generation/data/nonsymmetric_preds_set.pkl', 'rb') as f:
    nonsymmetric_preds_set = pickle.load(f)

def negate(s: str) -> str:
    return s[3:] if s[:3] == 'neg' else 'neg' + s

def generate_fact(x: Tuple[str, ...], b: int, rel: str) -> str:
    """Generate predicate using grounded subj and/or obj, polartiy, and relation"""
    return (rel + '(' + x[0] + ',' + x[1] + ').' if b == 0 else negate(rel) + '(' + x[0] + ',' + x[1] + ').')\
        if len(x) == 2 else (rel + '(' + x[0] + ').' if b == 0 else negate(rel) + '(' + x[0] + ').')


def solve_ASP(facts: str, rules: str) -> List[str]:
    """Generates list of satisfied predicates with facts and hard rules using ASP."""
    x = facts.lower() + ' ' + rules
    answers = ASP(x)
    d = list(list(answers)[0])
    if(d):
        if(len(d[0][1]) == 2):
            d = [x[0] + '(' + str(x[1][0]).capitalize() + ',' + str(x[1][1]).capitalize() + ')' for x in d
                 if x[1][0] != x[1][1]]
        else:
            d = [x[0] + '(' + str(x[1][0]).capitalize() + ')' for x in d]
    return d

def get_assignments(n_facts: int) -> List[List[int]]:
    """Returns a list of boolean assignments for predicate polarity"""
    n = 2**n_facts
    a = []
    for i in range(n):
        a.append([int(x) for x in bin(i)[2:].rjust(n_facts, '0')])
    return a


def my_shuffle(array):
    shuffle(array)
    return array


def symbolic2text(p: Dict[str, Any]) -> Dict[str, Any]:
    """Converts symbolic data to natural language"""
    # TODO Fix Path
    # Load pkl file containing rules in natural language
    with open('data_generation/data/rule2text.pkl', 'rb') as file:
        rule2text = pickle.load(file)

    f = [Triple(x) for x in p['facts'].split('.') if x]
    facts_sentences = [x.get_sentence(x.subject, x.object) for x in f]
    p['facts_sentence'] = " ".join(facts_sentences)

    h = Triple(p['hypothesis'])
    hyp_sent = h.get_sentence(h.subject, h.object)

    p['hypothesis_sentence'] = hyp_sent

    p['natural_rule'] = " ".join([rule2text[r] for r in p['rule']]) if isinstance(p['rule'], list) \
        else rule2text[p['rule']]
    # shuffle rules and facts
    p['context'] = ". ".join(my_shuffle((p['facts_sentence'] + ' ' + p['natural_rule'][:-1])
                             .replace(". ", "###").split("###"))) + '.'

    return p


def get_num(d, ll, s):
    return len([x for x in d if x[ll] == s])

def find_conflicting_facts(x: List[str]) -> bool:
    for xx in x:
        if(xx and negate(xx) in x):
            return True
    return False

def switch(triple: str) -> str:
    return Triple(triple).switch_subj_obj()
# TODO Update path
# TODO Customize RuntimeErrors
def solve_LPMLN(facts: str, rule: str, rule_support: float, complementary_rules: str) -> Dict[str, str]:
    """Solves program using LPMLN.

    Args:
        facts (str): facts of program
        rule (str): rule of program
        rule_support (float): rule support
        complementary_rules (str): complementary rules

    Raises:
        RuntimeError: [description]
        RuntimeError: [description]

    Returns:
        Dict[str, str]: dict where key is predicate and value is probability of being satisfied
    """
    lpmln_path = "data_generation/lpmln"
    facts = facts.replace('(', '("').replace(',', '","').replace(')', '")')
    weight = np.log((rule_support + 1e-308) / (1 - rule_support + 1e-308))

    all_program_relations = [Triple(x).relation for x in facts.split(".") if x] + Rule(rule).relations
    filtered_relations = sorted(set(all_program_relations), key=all_program_relations.index)
    predicate_hypothesis = ",".join(filtered_relations + [negate(x) for x in filtered_relations])

    # create temp file
    file_content = str(weight) + ' ' + rule + ' ' + complementary_rules + ' ' + facts
    print(file_content, file=open('temp', 'w'))

    # run command
    bashCommand = lpmln_path + '/lpmln_infer.py temp -q ' + predicate_hypothesis
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    output_, error = process.communicate()
    if(error):
        raise RuntimeError('Error in LPMLN Script')

    output = output_.decode('ascii') if type(output_) == bytes else output_
    output = str(output)
    if(len(output) < 3):
        raise RuntimeError('Empty Output in LPMLN Script')

    # parse output
    parsed_output = dict([x.replace(' ', "").split(":") for x in re.findall("(?<=\\n)(.*?)\\n", output) if x])
    return parsed_output


def solve_LPMLN_multi(facts: str, rules: List[str], rule_supports: List[float], complementary_rules: str,
                      timeout: float = None) -> Union[None, Dict[str, str]]:
    """ Solve program with facts and multiple rules.

    Args:
        facts (str): Facts
        rules (List[str]): List of rules
        rule_supports (List[float]): List of rule supports
        complementary_rules (str): complementary rules
        timeout (float, optional): Number of seconds for timeout. Defaults to None. Note: This is used when
        multiple rules are inputted and thus the solver might take some time for some examples. In this case,
        the solver will timeout and generate another example. NO ERRORS ARE THROWN HERE.

    Returns:
        Dict[str, str]: dict where key is predicate and value is probability of being satisfied
    """

    # TODO fix path
    lpmln_path = os.getcwd() + '/' + "lpmln"
    facts = facts.replace('(', '("').replace(',', '","').replace(')', '")')

    all_program_relations = [Triple(x).relation for x in facts.split(".") if x] + \
                            [r for rule in rules for r in Rule(rule).relations]
    filtered_relations = sorted(set(all_program_relations), key=all_program_relations.index)
    predicate_hypothesis = ",".join(filtered_relations + [negate(x) for x in filtered_relations])

    # create file
    weighted_rules = ''
    for i in range(len(rules)):
        weight = np.log((rule_supports[i] + 1e-308) / (1 - rule_supports[i] + 1e-308))
        weighted_rules += str(weight) + ' ' + rules[i] + ' '
    file_content = weighted_rules + ' ' + complementary_rules + ' ' + facts
    print(file_content, file=open('temp', 'w'))

    # run command
    bashCommand = lpmln_path + '/lpmln_infer.py ' + os.getcwd() + '/temp -q ' + predicate_hypothesis
    if(timeout):
        bashCommand = f'timeout {timeout} ' + bashCommand
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    output_, error = process.communicate()
    if(error):
        return None

    if(not output_):
        return None
    output = output_.decode('ascii') if type(output_) == bytes else output_
    output = str(output)
    if(len(output) < 3):
        return None

    parsed_output = dict([x.replace(' ', "").split(":") for x in re.findall("(?<=\\n)(.*?)\\n", output) if x])

    return parsed_output

def get_proba_lpmln(parsed_output: Dict[str, str], hypothesis: str) -> float:
    """Return probability given by LPMLN solver for a predicate"""
    # TODO Check if fixed
    # remove quotations
    if('"' not in hypothesis):
        hypothesis = hypothesis.replace('(', '("').replace(',', '","').replace(')', '")')

    if(hypothesis in parsed_output):
        prob = round(float(parsed_output[hypothesis]), 2)

    elif((Triple(hypothesis).relation in symmetric_preds_set
         or negate(Triple(hypothesis).relation) in symmetric_preds_set)
            and switch(hypothesis) in parsed_output and hypothesis in parsed_output):
        prob = round(max(float(parsed_output[switch(switch(hypothesis))]), float(parsed_output[switch(hypothesis)])), 2)

    # if symmetric
    elif((Triple(hypothesis).relation in symmetric_preds_set
         or negate(Triple(hypothesis).relation) in symmetric_preds_set)
         and switch(hypothesis) in parsed_output):
        prob = round(float(parsed_output[switch(hypothesis)]), 2)

    # if neg
    elif(negate(hypothesis) in parsed_output):
        prob = 1 - round(float(parsed_output[negate(hypothesis)]), 2)

    # if neg switch
    elif((Triple(hypothesis).relation in symmetric_preds_set
         or negate(Triple(hypothesis).relation) in symmetric_preds_set)
         and negate(switch(hypothesis)) in parsed_output):
        prob = 1 - round(float(parsed_output[negate(switch(hypothesis))]), 2)

    elif(switch(hypothesis) in parsed_output and hypothesis[:3] != 'neg'):
        prob = 1 - round(float(parsed_output[switch(hypothesis)]), 2)

    else:
        prob = 1 if hypothesis[:3] == 'neg' else 0

    return prob
import json
def dump_jsonl(file, filename, path):
    with open(path + filename, 'w') as f:
        for entry in file:
            json.dump(entry, f)
            f.write('\n')


def format_time(elapsed):
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))

def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

def confidence_accuracy(logits, labels, weights, threshold=0.05, verbose=False):
    # probs = torch.nn.functional.softmax(logits, dim=1)
    probs = softmax(logits, axis=1)
    # pred_weights = np.array([x[l].item() for x, l in zip(probs, labels)])
    pred_weights = np.array([x[1].item() for x in probs])  # get true prob
    abs_diff = np.abs(pred_weights - weights.cpu().numpy())
    if not verbose:
        return np.sum(abs_diff < threshold) / len(abs_diff)
    else:
        return probs, abs_diff

def get_hash(s):
    return int(hashlib.sha1(s.encode("utf-8")).hexdigest(), 16) % (10 ** 8)
