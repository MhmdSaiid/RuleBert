from .Triple import Triple
from random import sample
from typing import Dict, List, Tuple, Union
import pickle
import re


class Rule:
    # TODO add exaples of good and bad rules
    '''Assumptions:
        1) Variables in rules are designated by capital letters.
        2) The head of the rule is to the left.
        3) Operations not supported yet.
        4) Assuming no literals.
        5) :- is separator, head to the left
    '''
    def __init__(self, rule: str):

        Rule.add_rule_info(rule)
        self.rule = rule
        self.vars = sorted(list(set(re.findall("[A-Z]", rule))))  # get variables in the rule
        self.num_vars = len(self.vars)                            # get number of of variables

        h = re.findall(r"[a-zA-Z]+\(.*?,.*?\)", self.rule.split(':-')[0])  # split rule

        if(h):
            # rule with binary predicates
            self.head = Triple(h[0])
            self.body = [Triple(x) for x in re.findall(r"[a-zA-Z]+\(.*?,.*?\)", self.rule.split(':-')[1])]
        else:
            # rule with unary predicates
            self.head = Triple(re.findall(r"[a-zA-Z]+\(.*?\)", self.rule.split(':-')[0])[0])
            self.body = [Triple(x) for x in re.findall(r"[a-zA-Z]+\(.*?\)", self.rule.split(':-')[1])]

        self.triples = self.body + [self.head]
        self.positive_rule = False if self.head.triple[:3] == 'neg' else True
        self.relations = [x.relation for x in self.triples]

    def set_pool_dict(self, pool_list: List[List[str]]):
        assert(len(pool_list) == self.num_vars)
        self.pool_dict = dict(zip(self.vars, pool_list))

    def set_rule_support(self, w: Union[str, float]):
        self.rule_support = w

    @staticmethod
    def add_rule_info(rule: str):
        """Adds rule description and predicate metadata."""
        with open('data_generation/data/rule2text.pkl', 'rb') as f:
            rule2text = pickle.load(f)

        if(rule not in rule2text):
            text = input(f'Please enter the natural language description of the rule {rule}.')
            rule2text[rule] = text

            with open('data_generation/data/rule2text.pkl', 'wb') as f:
                pickle.dump(rule2text, f)

            print('Rule Description successfully added.')

            with open('data_generation/data/symmetric_preds_set.pkl', 'rb') as f:
                symmetric_preds_set = pickle.load(f)

            with open('data_generation/data/nonsymmetric_preds_set.pkl', 'rb') as f:
                nonsymmetric_preds_set = pickle.load(f)

            rule_relations = Rule(rule).relations

            for rel in rule_relations:
                if(rel not in symmetric_preds_set and rel not in nonsymmetric_preds_set):
                    text = ''
                    while(text not in ['y', 'n']):
                        text = input(f'Is the relation "{rel}" symmetric? (y/n)')
                        text = text.lower()
                        if(text == 'y'):
                            symmetric_preds_set.add(rel)
                            with open('data_generation/data/symmetric_preds_set.pkl', 'wb') as f:
                                pickle.dump(symmetric_preds_set, f)

                        elif(text == 'n'):
                            nonsymmetric_preds_set.add(rel)
                            with open('data_generation/data/nonsymmetric_preds_set.pkl', 'wb') as f:
                                pickle.dump(nonsymmetric_preds_set, f)

            print('Successfully added relation info.')

    def generate_atom_space(self, name_pool: Dict[str, List[str]]) -> List[str]:
        """Generate rule atom space"""
        rule_atom_space = [t.generate_atom_space(name_pool) for t in self.triples]
        rule_atom_space = [item for sublist in rule_atom_space for item in sublist]
        rule_atom_space = sorted(set(rule_atom_space), key=rule_atom_space.index)
        return rule_atom_space

    def random_sample(self, name_pool_dict: Dict[str, List[str]]) -> List[str]:
        """Generate random rule-body predicates satisfying the rule"""
        var_dict = {k: sample(name_pool_dict[k], 1)[0] for k in self.vars}
        rs = []
        for x in self.body:
            triple_vars = x.vars
            g = x.ground(var_dict[triple_vars[0]], var_dict[triple_vars[1]])\
                if len(triple_vars) == 2 else x.ground(var_dict[triple_vars[0]])
            rs.append(g + '.')
        return rs

    def random_sample_with_assignment(self, name_pool_dict: Dict[str, List[str]]) \
            -> Tuple[List[str], Dict[str, str]]:

        """Generate random rule-body predicates satisfying the rule making sure there
           are no duplicate assignments."""

        var_dict = {k: sample(name_pool_dict[k], 1)[0] for k in self.vars}

        # make sure we do not have duplicate assignments
        while(len(set(list(var_dict.values()))) != self.num_vars):
            var_dict = {k: sample(name_pool_dict[k], 1)[0] for k in self.vars}

        rs: List[str] = []
        for x in self.body:
            triple_vars = x.vars
            g = x.ground(var_dict[triple_vars[0]], var_dict[triple_vars[1]])\
                if len(triple_vars) == 2 else x.ground(var_dict[triple_vars[0]])
            rs.append(g + '.')
        return rs, var_dict

    def __str__(self):
        body_str = '\n'.join([x.__str__() for x in self.body])
        return f"Rule: {self.rule}\n\nWeight: {self.w}\n\nHead:{self.head.__str__()}\n\nBody:\n{body_str}\n\n\
                 Rule Type: {self.rule_type}"
