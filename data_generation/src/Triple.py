from itertools import product
from typing import Dict, List
import re
import pickle

class Triple:
    # TODO update description with assumptions
    """ Class for Triple functions"""

    # assuming no literals for now
    # Assume variables with A-Z

    def __init__(self, triple: str):

        self.triple = triple
        self.relation = re.findall(r"([a-zA-Z<>=]+)\(", self.triple)[0]
        self.subject_list = re.findall(r"\((.*?),", self.triple)
        # check if binary predicate
        if(self.subject_list):
            self.subject = self.subject_list[0]
            self.object = re.findall(r"\(.*?,(.*?)\)", self.triple)[0]

        else:
            # unary predicate
            self.subject = re.findall(r"\((.*?)\)", self.triple)[0]
            self.object = None

        self.vars = re.findall(r"[A-Z]", triple)

    @staticmethod
    def negate(s: str) -> str:
        """Negates an input triple"""
        return s[3:] if s[:3] == 'neg' else 'neg' + s

    def ground(self, subj: str, obj: str = None) -> str:
        """Grounds a binary/unary predicate"""
        return self.relation + '(' + subj + ',' + obj + ')' if(self.object) else self.relation + '(' + subj + ')'

    def generate_atom_space(self, name_pool_dict: Dict[str, List[str]]) -> List[str]:
        """Generate space of all atoms given lists of subjects and objects for input predicate"""
        subj_pool = name_pool_dict[self.subject]
        obj_pool = name_pool_dict[self.object]
        # TODO Can this be optimized
        name_perms = list(product(subj_pool, obj_pool))
        pos_gr_triples = [self.ground(x[0], x[1]) for x in name_perms]
        atom_space = pos_gr_triples + [self.negate(x) for x in pos_gr_triples]
        return atom_space

    def get_sentence(self, grounded_subject, grounded_object=None):

        # COMPLETED: Replace by input structure sentence: The !!R!! of !!S!! is !!O!!.Inlcude negative

        with open('data_generation/data/rel2text.pkl', 'rb') as f:
            rel2text = pickle.load(f)

        if self.relation in rel2text:
            sent = rel2text[self.relation]
        else:
            if self.relation[:3]!='neg':
                sent = f"The {self.relation} of !!S!! is !!O!!." if self.object else f"!!S!! is {self.relation}."
            else:
                sent = f"The {self.relation[3:]} of !!S!! is not !!O!!." if self.object else f"!!S!! is not {self.relation[3:]}."

        sent = sent.replace("!!S!!",grounded_subject)

        if self.object:
            sent = sent.replace("!!O!!",grounded_object)
        return sent

        # if('connected' in self.relation):
        #     return self.subject + ' is not connected to ' + self.object + '.' \
        #         if self.relation[:3] == 'neg' \
        #         else self.subject + ' is connected to ' + self.object + '.'

        # if('friends' in self.relation):
        #     return self.subject + ' is not friends with ' + self.object + '.' \
        #         if self.relation[:3] == 'neg' \
        #         else self.subject + ' is friends with ' + self.object + '.'

        # if('engaged' in self.relation):
        #     return self.subject + ' is not engaged to ' + self.object + '.' \
        #         if self.relation[:3] == 'neg' \
        #         else self.subject + ' is engaged to ' + self.object + '.'

        # if('close' in self.relation):
        #     return self.subject + ' is not close to ' + self.object + '.' \
        #         if self.relation[:3] == 'neg' \
        #         else self.subject + ' is close to ' + self.object + '.'

        # if('dating' in self.relation):
        #     return self.subject + ' is not dating ' + self.object + '.' \
        #         if self.relation[:3] == 'neg' \
        #         else self.subject + ' is dating ' + self.object + '.'

        # if('married' in self.relation):
        #     return self.subject + ' is not married to ' + self.object + '.' \
        #         if self.relation[:3] == 'neg' \
        #         else self.subject + ' is married to ' + self.object + '.'

        # if('related' in self.relation):
        #     return self.subject + ' is not related to ' + self.object + '.' \
        #         if self.relation[:3] == 'neg' \
        #         else self.subject + ' is related to ' + self.object + '.'
        # if('studiesat' in self.relation):
        #     return self.subject + ' does not study at ' + self.object + '.' \
        #         if self.relation[:3] == 'neg' \
        #         else self.subject + ' studies at ' + self.object + '.'

        # if('worksat' in self.relation):
        #     return self.subject + ' does not work at ' + self.object + '.' \
        #         if self.relation[:3] == 'neg' \
        #         else self.subject + ' works at ' + self.object + '.'
        # if('bestfriend' in self.relation):
        #     return 'The best friend of ' + self.subject + ' is not ' + self.object + '.' \
        #         if self.relation[:3] == 'neg' \
        #         else 'The best friend of ' + self.subject + ' is ' + self.object + '.'

        # if(self.object):
        #     return 'The ' + self.relation[3:] + ' of ' + self.subject + ' is not ' + self.object + '.' \
        #         if self.relation[:3] == 'neg' \
        #         else 'The ' + self.relation + ' of ' + self.subject + ' is ' + self.object + '.'
        # else:
        #     return self.subject + ' is not ' + self.relation[3:] + '.' \
        #         if self.relation[:3] == 'neg' \
        #         else self.subject + ' is ' + self.relation + '.'

    def switch_subj_obj(self) -> str:
        """Switches subject and object in a triple"""
        return self.relation + '(' + self.object + ',' + self.subject + ')'

    def __str__(self):
        return f"Triple: {self.triple}\nRelation: {self.relation}\nSubject: {self.subject}\nObject: {self.object}"
