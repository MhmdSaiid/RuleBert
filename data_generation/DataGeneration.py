from src.utils import negate, generate_fact, solve_ASP, solve_LPMLN, Rule, get_assignments, get_proba_lpmln, \
    find_conflicting_facts, Triple, switch, symbolic2text, solve_LPMLN_multi, dump_jsonl
from tqdm import tqdm
from random import randint, sample, random
from itertools import product
from copy import deepcopy
import pickle
import time
import argparse
import json
import ast
from pathlib import Path
from src.utils import get_hash
from sklearn.model_selection import train_test_split
# COMPLETED Add Chaining
# COMPLETED Add code for running exps
# TODO Type Hinting
# COMPLETED Add CLI
# TODO add __init__.py
# TODO Remove printed output from LPMLN

class OneRuleDataGenerator:
    def __init__(self, rule, pool_list, max_num_facts, rule_support=None):
        """Class to generate theories for positve/negative hard/soft rules.

        Args:
            rule (str): Rule with head in the lhs example: negspouse(A,B) :- child(A,B).
            pool_list (List[List[str]]): list of names for each variable in the rule
            max_num_facts (int): maximum number of facts in a theory
            rule_support (float, optional): Support of the rule. For hard rules, set to None.
        """

        self.rule = rule
        self.pool_list = pool_list
        self.max_num_facts = max_num_facts
        self.rule_support = rule_support
        self.lpmln = True if self.rule_support else False

    def generate_theories(self, num=500, negation_as_failure=True, complementary_rules='', p_bar=True):

        """Class to generate theories for positve/negative hard/soft rules.

        Args:
            num (int, optional): number of theories per meta. Defaults to 500.
            negation_as_failure (bool, optional): when set to False, unsatisifed atoms are labeled as 'UNK'.
                                                  Defaults to True.
            complementary_rules (str, optional): a string of rules to include in the LPMLN program,
                                                 ex:   :-spouse(A,A). Defaults to ''.
            p_bar (bool, optional):  progress bar. Defaults to True.

        Returns:
            true_theories (List[Dict[str, Any]]): Generated true theories
            false_theories (List[Dict[str, Any]]): Generated false theories
        """

        # load set of symmetric predicates
        with open('data_generation/data/symmetric_preds_set.pkl', 'rb') as f:
            symmetric_preds_set = pickle.load(f)

        rp = Rule(self.rule)  # Rule Parser
        rp.set_pool_dict(self.pool_list)

        all_n_pools = []
        true_theories = []
        false_theories = []

        c = 0
        if (p_bar):
            bar = tqdm(total=c)

        while(c < num):

            num_facts = randint(1, self.max_num_facts)  # get random number of facts
            num_names = randint(rp.num_vars, self.max_num_facts + 1)  # get random set of names

            pool_sample = {}
            for var in rp.vars:
                pool_sample[var] = sample(rp.pool_dict[var], num_names)  # sample from each pool

            # list of all name combinations for each triple in body according to vars in triple
            for h in rp.body:
                all_n_pools.append(list(product(rp.pool_dict[h.subject], rp.pool_dict[h.object])))

            all_atoms = rp.generate_atom_space(pool_sample)  # generate all atom space

            # get pool of relations
            rel_pool = [x.relation for x in rp.body]  # get relations from rule body
            rel_pool = list(product(rel_pool, repeat=num_facts))  # get combinations
            br = [x.relation for x in rp.body]

            # heuristic to encourage having predicates of the body of the rule.
            rel_pool = [y for y in rel_pool if sum([y.count(x) for x in set(br)]) >= round(0.8 * num_facts)]

            #  list of all polarity assignments: 3 facts --> 2^3=8 assignments
            assignments = get_assignments(num_facts)[:-1]
            rel = sample(rel_pool, 1)[0]  # sample from relation pool
            assignmnet = sample(assignments, 1)[0]  # sample from assignment pool

            facts = []
            # find facts per assignment
            for aa, r in zip(assignmnet, rel):
                pool_index = [x.relation for x in rp.body].index(r)
                pp = sample(all_n_pools[pool_index], 1)[0]
                facts.append(generate_fact(pp, aa, r))

            # remove some facts
            preds2sample = len(facts) - len(rp.body)
            preds2sample = preds2sample if preds2sample > 0 else len(facts)

            facts_aug_split = sample(facts, preds2sample) + \
                rp.random_sample(pool_sample)  # add predicates satisfying rule

            facts = " ".join(facts)
            facts_aug = " ".join(facts_aug_split)

            # redo fact generation if there were conflicting facts or rule was not trigerred with facts_aug
            while(find_conflicting_facts(facts_aug_split)
                  or not [x for x in solve_ASP(facts_aug, self.rule) if rp.head.relation in x]):

                rel = sample(rel_pool, 1)[0]
                assignmnet = sample(assignments, 1)[0]

                facts = []
                for aa, r in zip(assignmnet, rel):
                    pool_index = [x.relation for x in rp.body].index(r)
                    pp = sample(all_n_pools[pool_index], 1)[0]
                    facts.append(generate_fact(pp, aa, r))

                preds2sample = len(facts) - len(rp.body)
                preds2sample = preds2sample if preds2sample > 0 else len(facts)
                facts_aug_split = sample(facts, preds2sample) + rp.random_sample(pool_sample)

                facts = " ".join(facts)
                facts_aug = " ".join(facts_aug_split)

            # solve
            if(self.lpmln):
                lpmln_prob = solve_LPMLN(facts_aug, self.rule, self.rule_support, complementary_rules)

            asp_answer_aug = solve_ASP(facts_aug, self.rule)

            asp_answer_w_head = [x for x in asp_answer_aug if rp.head.relation in x]

            # find head_hypotheses
            all_head_hyps = [x for x in all_atoms if rp.head.relation in x
                             and Triple(x).object != Triple(x).subject]

            if(rp.positive_rule):
                all_head_hyps = [x for x in all_head_hyps if 'neg' not in x]

            # satisfied rule head
            rconc = asp_answer_w_head[0]
            rconc_meta = 'rconc'

            # check if symmetric
            if((Triple(rconc).relation in symmetric_preds_set
                or negate(Triple(rconc).relation) in symmetric_preds_set)
                    and random() > 0.5):

                # switch 50% of the time
                rconc = switch(rconc)
                rconc_meta = 'switch_rconc'

            p_theory = {'facts': facts_aug,
                        'hypothesis': rconc,
                        'rule': self.rule,
                        'output': True,
                        'solution': asp_answer_aug,
                        'meta': rconc_meta}
            h = get_hash(p_theory['facts'] + " ".join(p_theory['rule']) + p_theory['hypothesis'] + p_theory['meta'])
            p_theory['id'] = f'Single_{h}'

            if(self.lpmln):
                p_theory['hyp_weight'] = get_proba_lpmln(lpmln_prob, rconc)
                p_theory['rule_support'] = self.rule_support
                p_theory['solution'] = lpmln_prob

            true_theories.append(symbolic2text(p_theory))

            # check if non-symmetric and switch 50% of the time
            if(Triple(rconc).relation not in symmetric_preds_set
               and negate(Triple(rconc).relation) not in symmetric_preds_set and random() > 0.5):

                f_theory = {'facts': facts_aug,
                            'hypothesis': switch(rconc),
                            'rule': self.rule,
                            'output': False,
                            'solution': asp_answer_aug,
                            'meta': 'switch_rconc'}
                h = get_hash(f_theory['facts'] + " ".join(f_theory['rule']) + f_theory['hypothesis'] + f_theory['meta'])
                f_theory['id'] = f'Single_{h}'
                if(self.lpmln):
                    f_theory['hyp_weight'] = get_proba_lpmln(lpmln_prob, switch(rconc))
                    f_theory['rule_support'] = self.rule_support
                    f_theory['solution'] = lpmln_prob

                false_theories.append(symbolic2text(f_theory))

            else:
                # negate satisfied positive rule head
                f_theory = {'facts': facts_aug,
                            'hypothesis': negate(rconc),
                            'rule': self.rule,
                            'output': False,
                            'solution': asp_answer_aug,
                            'meta': 'inv_rconc'}
                h = get_hash(f_theory['facts'] + " ".join(f_theory['rule']) + f_theory['hypothesis'] + f_theory['meta'])
                f_theory['id'] = f'Single_{h}'
                if(self.lpmln):
                    f_theory['hyp_weight'] = get_proba_lpmln(lpmln_prob, negate(rconc))
                    f_theory['rule_support'] = self.rule_support
                    f_theory['solution'] = lpmln_prob

                false_theories.append(symbolic2text(f_theory))

            # find unsatisfied positive rule head
            unsat_head_sample = sample(list(set(all_head_hyps).difference(asp_answer_aug)), 1)[0]

            # find unsatisifed positive fact
            unsat_fact_atoms = [x for x in all_atoms if x not in asp_answer_aug and negate(rp.head.relation) not in x
                                and rp.head.relation not in x and 'neg' not in x]

            h = unsat_head_sample if rp.positive_rule else negate(unsat_head_sample)
            f_theory = {'facts': facts_aug,
                        'hypothesis': h,
                        'rule': self.rule,
                        'output': False if negation_as_failure else 'UNK',
                        'solution': asp_answer_aug,
                        'meta': 'unsat_rconc' if rp.positive_rule else 'inv_unsat_rconc'}
            hash = get_hash(f_theory['facts'] + " ".join(f_theory['rule']) + f_theory['hypothesis'] + f_theory['meta'])
            f_theory['id'] = f'Single_{hash}'
            if(self.lpmln):
                f_theory['hyp_weight'] = get_proba_lpmln(lpmln_prob, h)
                f_theory['rule_support'] = self.rule_support
                f_theory['solution'] = lpmln_prob

            false_theories.append(symbolic2text(f_theory))

            h = negate(unsat_head_sample) if rp.positive_rule else unsat_head_sample
            p_theory = {'facts': facts_aug,
                        'hypothesis': h,
                        'rule': self.rule,
                        'output': True if negation_as_failure else 'UNK',
                        'solution': asp_answer_aug,
                        'meta': 'inv_unsat_rconc' if rp.positive_rule else 'unsat_rconc'}
            hash = get_hash(p_theory['facts'] + " ".join(p_theory['rule']) + p_theory['hypothesis'] + p_theory['meta'])
            p_theory['id'] = f'Single_{hash}'
            if(self.lpmln):
                p_theory['hyp_weight'] = get_proba_lpmln(lpmln_prob, h)
                p_theory['rule_support'] = self.rule_support
                p_theory['solution'] = lpmln_prob

            true_theories.append(symbolic2text(p_theory))

            if(unsat_fact_atoms):
                unsat_fact_atom = sample(unsat_fact_atoms, 1)[0]

                f_theory = {'facts': facts_aug,
                            'hypothesis': unsat_fact_atom,
                            'rule': self.rule,
                            'output': False if negation_as_failure else 'UNK',
                            'solution': asp_answer_aug,
                            'meta': 'unsat_fact'}
                h = get_hash(f_theory['facts'] + " ".join(f_theory['rule']) + f_theory['hypothesis'] + f_theory['meta'])
                f_theory['id'] = f'Single_{h}'
                if(self.lpmln):
                    f_theory['hyp_weight'] = get_proba_lpmln(lpmln_prob, unsat_fact_atom)
                    f_theory['rule_support'] = self.rule_support
                    f_theory['solution'] = lpmln_prob

                false_theories.append(symbolic2text(f_theory))

                p_theory = {'facts': facts_aug,
                            'hypothesis': negate(unsat_fact_atom),
                            'rule': self.rule,
                            'output': True if negation_as_failure else 'UNK',
                            'solution': asp_answer_aug,
                            'meta': 'inv_unsat_fact'}
                h = get_hash(p_theory['facts'] + " ".join(p_theory['rule']) + p_theory['hypothesis'] + p_theory['meta'])
                p_theory['id'] = f'Single_{h}'
                if(self.lpmln):
                    p_theory['hyp_weight'] = get_proba_lpmln(lpmln_prob, negate(unsat_fact_atom))
                    p_theory['rule_support'] = self.rule_support
                    p_theory['solution'] = lpmln_prob

                true_theories.append(symbolic2text(p_theory))

            # sample positive fact
            if(self.lpmln):
                sat_pos_facts = [x for x in list(lpmln_prob.keys()) if negate(rp.head.relation) not in x
                                 and rp.head.relation not in x and lpmln_prob[x] == '1.0']

                if(not sat_pos_facts):
                    sat_pos_facts = [x for x in asp_answer_aug if negate(rp.head.relation) not in x
                                     and rp.head.relation not in x]

            else:
                sat_pos_facts = [x for x in asp_answer_aug if negate(rp.head.relation) not in x
                                 and rp.head.relation not in x]

            if(sat_pos_facts):

                sat_fact = sample(sat_pos_facts, 1)[0]
                sat_fact_meta = 'fact'

                # if symmetric fact then switch subj and obj 50% of the time
                if((Triple(sat_fact).relation in symmetric_preds_set
                    or negate(Triple(sat_fact).relation) in symmetric_preds_set)
                        and random() > 0.5):

                    sat_fact = switch(sat_fact)
                    sat_fact_meta = 'switch_fact'

                p_theory = {'facts': facts_aug,
                            'hypothesis': sat_fact,
                            'rule': self.rule,
                            'output': True,
                            'solution': asp_answer_aug,
                            'meta': sat_fact_meta}
                h = get_hash(p_theory['facts'] + " ".join(p_theory['rule']) + p_theory['hypothesis'] + p_theory['meta'])
                p_theory['id'] = f'Single_{h}'
                if(self.lpmln):
                    p_theory['hyp_weight'] = get_proba_lpmln(lpmln_prob, sat_fact)
                    p_theory['rule_support'] = self.rule_support
                    p_theory['solution'] = lpmln_prob

                true_theories.append(symbolic2text(p_theory))

                #  if not symmetric then switch
                if(Triple(sat_fact).relation not in symmetric_preds_set
                   and negate(Triple(sat_fact).relation) not in symmetric_preds_set
                   and random() > 0.5):

                    f_theory = {'facts': facts_aug,
                                'hypothesis': switch(sat_fact),
                                'rule': self.rule,
                                'output': False,
                                'solution': asp_answer_aug,
                                'meta': 'switch_fact'}
                    h = get_hash(f_theory['facts'] + " ".join(f_theory['rule']) + f_theory['hypothesis']
                                 + f_theory['meta'])
                    f_theory['id'] = f'Single_{h}'
                    if(self.lpmln):
                        f_theory['hyp_weight'] = get_proba_lpmln(lpmln_prob, switch(sat_fact))
                        f_theory['rule_support'] = self.rule_support
                        f_theory['solution'] = lpmln_prob

                    false_theories.append(symbolic2text(f_theory))

                else:
                    f_theory = {'facts': facts_aug,
                                'hypothesis': negate(sat_fact),
                                'rule': self.rule,
                                'output': False,
                                'solution': asp_answer_aug,
                                'meta': 'inv_fact'}
                    h = get_hash(f_theory['facts'] + " ".join(f_theory['rule']) + f_theory['hypothesis']
                                 + f_theory['meta'])
                    f_theory['id'] = f'Single_{h}'
                    if(self.lpmln):
                        f_theory['hyp_weight'] = get_proba_lpmln(lpmln_prob, negate(sat_fact))
                        f_theory['rule_support'] = self.rule_support
                        f_theory['solution'] = lpmln_prob

                    false_theories.append(symbolic2text(f_theory))
            c += 1

            if (p_bar):
                bar.update(1)

        return true_theories, false_theories


class MultiRuleDataGenerator:
    def __init__(self, rules, pool_lists, max_num_facts, rule_supports=None):
        """Class to generate theories for the union of positve/negative hard/soft rules.

        Args:
            rules (List[str]): List of rules with head in the lhs example: negspouse(A,B) :- child(A,B).
                            All rules should have the same predicate/neg(predicate) as the head.
            pool_lists (List[List[str]]): list of lists of names for each variable
            max_num_facts (int): maximum number of facts in a theory
            rule_supports (List[float], optional): List of rule supports. Set to None for hard rules.
        """
        self.rules = rules
        self.pool_lists = pool_lists
        self.max_num_facts = max_num_facts
        self.rule_supports = rule_supports
        self.lpmln = True if self.rule_supports else False

    def generate_theories(self, num=500, negation_as_failure=True, complementary_rules='', p_bar=True):
        """Class to generate theories for the union of positve/negative hard/soft rules sharing same rule-head predicate.

        Args:
            num (int, optional): number of theories per meta. Defaults to 500.
            negation_as_failure (bool, optional): when set to False, we label unsatisifed atoms as 'UNKNOWN'.
                                                  Defaults to True.
            complementary_rules (str, optional): a string of rules to include in the LPMLN program,
                                                 ex:   :-spouse(A,A).. Defaults to ''.
            p_bar (bool, optional): progress bar. Defaults to True.

        Returns:
            true_theories (List[Dict[str, Any]]): Generated true theories
            false_theories (List[Dict[str, Any]]): Generated false theories
        """

        #  set of symmetric predicates
        with open('data_generation/data/symmetric_preds_set.pkl', 'rb') as f:
            symmetric_preds_set = pickle.load(f)

        # get rule masks: binary indicators of which rules to trigger
        rule_masks = [x for x in get_assignments(len(self.rules)) if sum(x) > 1]

        rps = [Rule(rule) for rule in self.rules]  # parse rules

        for i, rp in enumerate(rps):
            rp.set_pool_dict(self.pool_lists[i])   # set pool dictionary

        rp_head = rps[0].head.triple  # get rule head triple
        # pred and neg(pred) can not occur together
        complementary_rules += " :- " + rp_head + "," + negate(rp_head) + "."
        rule_head_relation = rps[0].head.relation   # get rule head relation
        rule_head_relation = rule_head_relation[3:] if rule_head_relation[:3] == 'neg' else rule_head_relation

        all_n_pools = []
        true_theories = []
        false_theories = []

        c = 0
        if (p_bar):
            bar = tqdm(total=c)

        min_num_facts = sum([len(rp.body) for rp in rps])
        max_num_vars = max([rp.num_vars for rp in rps])

        while(c < num):

            num_facts = randint(min_num_facts, self.max_num_facts)
            num_names = randint(max_num_vars, self.max_num_facts + 1)

            pool_sample = []
            for rp in rps:
                rp_pool_sample = {}
                for var in rp.vars:
                    #  sample from each pool
                    rp_pool_sample[var] = sample(rp.pool_dict[var], num_names)
                pool_sample.append(rp_pool_sample)

            # list of all name combinations for each triple in body
            for rp in rps:
                rp_n_pool = []
                for h in rp.body:
                    # TODO adjust generator
                    rp_n_pool.append(list(product(rp.pool_dict[h.subject], rp.pool_dict[h.object])))
                all_n_pools.append(rp_n_pool)

            # get pool of relations
            rel_pools = []
            for rp in rps:
                rel_pool = [x.relation for x in rp.body]  # get relations from rule body
                rel_pool = list(product(rel_pool, repeat=num_facts))
                br = [x.relation for x in rp.body]
                rel_pool = [y for y in rel_pool if sum([y.count(x) for x in set(br)]) >= round(0.8 * num_facts)]
                rel_pools.append(rel_pool)
            self.rel_pools = rel_pools
            # list of all polarity assignments: 3 facts --> 2^3=8 assignments
            assignments = get_assignments(num_facts)[:-1]

            # sample from relation pool
            rels = []
            for i in range(len(rps)):
                rel = sample(rel_pools[i], 1)[0]
                rels.append(rel)

            #  sample from assignment pool
            assignmnet = []
            for i in range(len(rps)):
                assignmnet.append(sample(assignments, 1)[0])

            facts = []
            # find facts per assignment
            for i in range(len(rps)):
                for aa, r in zip(assignmnet[i], rels[i]):
                    pool_index = [x.relation for x in rps[i].body].index(r)
                    pp = sample(all_n_pools[i][pool_index], 1)[0]
                    facts.append(generate_fact(pp, aa, r))

            facts = sample(facts, num_facts)
            #  remove some facts
            preds2sample = len(facts) - sum([len(rp.body) for rp in rps])
            preds2sample = preds2sample if preds2sample > 0 else len(facts)

            ind_rule_with_max_vars = rps.index(max(rps, key=lambda rp: len(rp.body)))

            # find rule with max variables and sample from pool
            pp = {k: sample(pool_sample[ind_rule_with_max_vars][k], 1) for k in rps[ind_rule_with_max_vars].vars}

            for rule_mask in rule_masks:

                rules2fire = [r for i, r in enumerate(self.rules) if rule_mask[i]]
                meta_supplement = "_".join(['r' + str(i + 1) for i, v in enumerate(rule_mask) if v == 1])
                facts_aug_split = sample(facts, preds2sample)
                for i, rp in enumerate(rps):
                    if(rule_mask[i]):
                        facts_aug_split.extend(rp.random_sample(pp))  # add rule-trigerring facts

                facts_aug = " ".join(facts_aug_split)

                # redo fact generation if there were conflicting facts or rule was not trigerred.
                while(find_conflicting_facts(facts_aug_split)
                      or not [x for x in solve_ASP(facts_aug, " ".join(self.rules)) if rps[0].head.relation in x]):

                    # sample from relation pool
                    rels = []
                    for i in range(len(rps)):
                        rel = sample(rel_pools[i], 1)[0]
                        rels.append(rel)

                    assignmnet = []
                    for i in range(len(rps)):
                        assignmnet.append(sample(assignments, 1)[0])

                    facts = []
                    for i in range(len(rps)):
                        for aa, r in zip(assignmnet[i], rels[i]):
                            pool_index = [x.relation for x in rps[i].body].index(r)
                            pp = sample(all_n_pools[i][pool_index], 1)[0]
                            facts.append(generate_fact(pp, aa, r))
                    facts = sample(facts, num_facts)

                    preds2sample = len(facts) - sum([len(rp.body) for rp in rps])
                    preds2sample = preds2sample if preds2sample > 0 else len(facts)

                    ind_rule_with_max_vars = rps.index(max(rps, key=lambda rp: len(rp.body)))

                    pp = {k: sample(pool_sample[ind_rule_with_max_vars][k], 1)
                          for k in rps[ind_rule_with_max_vars].vars}

                    facts_aug_split = sample(facts, preds2sample)
                    for i, rp in enumerate(rps):
                        if(rule_mask[i]):
                            facts_aug_split.extend(rp.random_sample(pp))

                    facts_aug = " ".join(facts_aug_split)

                #  solve
                if(self.lpmln):
                    lpmln_prob = solve_LPMLN_multi(facts_aug, self.rules, self.rule_supports, complementary_rules, 3)
                    if(not lpmln_prob):
                        continue

                asp_answer_aug = solve_ASP(facts_aug, " ".join(self.rules))

                #  satisfied rule head
                pos_rconc = [x for x in asp_answer_aug if rule_head_relation in x and 'neg' not in x
                             and pp['A'][0] in x and pp['B'][0] in x]  # subj and obj should be included

                neg_rconc = [x for x in asp_answer_aug if rule_head_relation in x and 'neg' in x
                             and pp['A'][0] in x and pp['B'][0] in x]

                pos_rconc = pos_rconc[0] if pos_rconc else pos_rconc
                neg_rconc = neg_rconc[0] if neg_rconc else neg_rconc
                pos_output = True
                neg_output = True
                pos_meta_prefix = ''
                neg_meta_prefix = ''

                if(not neg_rconc):
                    neg_rconc = negate(pos_rconc)
                    neg_output = False
                    neg_meta_prefix = 'inv_'

                if(not pos_rconc):
                    pos_rconc = negate(neg_rconc)
                    pos_output = False
                    pos_meta_prefix = 'inv_'

                pos_rconc_meta = pos_meta_prefix + 'rconc' + '_' + meta_supplement
                neg_rconc_meta = neg_meta_prefix + 'rconc' + '_' + meta_supplement

                # pos_rconc
                # check if symmetric
                if((Triple(pos_rconc).relation in symmetric_preds_set
                    or negate(Triple(pos_rconc).relation) in symmetric_preds_set)
                        and random() > 0.5):

                    # switch 50% of the time
                    pos_rconc = switch(pos_rconc)
                    pos_rconc_meta = 'switch_' + pos_rconc_meta

                theory = {'facts': facts_aug,
                          'hypothesis': pos_rconc,
                          'rule': self.rules,
                          'output': pos_output,
                          'solution': asp_answer_aug,
                          'meta': pos_rconc_meta,
                          'satisifed_rules': rules2fire}
                h = get_hash(theory['facts'] + " ".join(theory['rule']) + theory['hypothesis'] + theory['meta'])
                theory['id'] = f'Multi_{h}'
                if(self.lpmln):
                    theory['hyp_weight'] = get_proba_lpmln(lpmln_prob, pos_rconc)
                    theory['rule_support'] = self.rule_supports
                    theory['solution'] = lpmln_prob

                true_theories.append(symbolic2text(theory)) if theory['output'] \
                    else false_theories.append(symbolic2text(theory))

                # neg rconc
                if((Triple(neg_rconc).relation in symmetric_preds_set
                    or negate(Triple(neg_rconc).relation) in symmetric_preds_set)
                        and random() > 0.5):

                    # switch 50% of the time
                    neg_rconc = switch(neg_rconc)
                    neg_rconc_meta = 'switch_' + neg_rconc_meta

                theory = {'facts': facts_aug,
                          'hypothesis': neg_rconc,
                          'rule': self.rules,
                          'output': neg_output,
                          'solution': asp_answer_aug,
                          'meta': neg_rconc_meta,
                          'satisifed_rules': rules2fire}
                h = get_hash(theory['facts'] + " ".join(theory['rule']) + theory['hypothesis'] + theory['meta'])
                theory['id'] = f'Multi_{h}'
                if(self.lpmln):
                    theory['hyp_weight'] = get_proba_lpmln(lpmln_prob, neg_rconc)
                    theory['rule_support'] = self.rule_supports
                    theory['solution'] = lpmln_prob
                    # theory['output']=True if theory['hyp_weight']>0.5 else False

                true_theories.append(symbolic2text(theory)) if theory['output'] \
                    else false_theories.append(symbolic2text(theory))

                # check if non-symmetric and switch
                if(Triple(pos_rconc).relation not in symmetric_preds_set
                   and negate(Triple(pos_rconc).relation) not in symmetric_preds_set):

                    f_theory = {'facts': facts_aug,
                                'hypothesis': switch(pos_rconc),
                                'rule': self.rules,
                                'output': False,
                                'solution': asp_answer_aug,
                                'meta': pos_rconc_meta,
                                'satisifed_rules': rules2fire}
                    h = get_hash(f_theory['facts'] + " ".join(f_theory['rule']) + f_theory['hypothesis']
                                 + f_theory['meta'])
                    f_theory['id'] = f'Multi_{h}'
                    if(self.lpmln):
                        f_theory['hyp_weight'] = get_proba_lpmln(lpmln_prob, switch(pos_rconc))
                        f_theory['rule_support'] = self.rule_supports
                        f_theory['solution'] = lpmln_prob

                    false_theories.append(symbolic2text(f_theory))

                # check if non-symmetric and switch
                if(Triple(neg_rconc).relation not in symmetric_preds_set
                   and negate(Triple(neg_rconc).relation) not in symmetric_preds_set):

                    f_theory = {'facts': facts_aug,
                                'hypothesis': switch(neg_rconc),
                                'rule': self.rules,
                                'output': False,
                                'solution': asp_answer_aug,
                                'meta': neg_rconc_meta,
                                'satisifed_rules': rules2fire}
                    h = get_hash(f_theory['facts'] + " ".join(f_theory['rule']) + f_theory['hypothesis']
                                 + f_theory['meta'])
                    f_theory['id'] = f'Multi_{h}'
                    if(self.lpmln):
                        f_theory['hyp_weight'] = get_proba_lpmln(lpmln_prob, switch(neg_rconc))
                        f_theory['rule_support'] = self.rule_supports
                        f_theory['solution'] = lpmln_prob

                    false_theories.append(symbolic2text(f_theory))

            c += 1

            if (p_bar):
                bar.update(1)

        return true_theories, false_theories

class ChainedRulesDataGenerator:
    def __init__(self, rules, pool_lists, rule_supports=None):
        """Class to generate theories for chained positve/negative hard/soft rules.

        Args:
            rules (List[str]): List of rules to chain from
            pool_lists (List[List[str]]): List of list of name pools for each rule
            rule_supports (List[float], optional): List of rule supports. Defaults to None for hard rule.
        """

        self.rules = rules
        self.pool_lists = pool_lists
        self.rule_supports = rule_supports
        self.lpmln = True if self.rule_supports else False

    def generate_theories(self, num=500, chaining_depth=5, negation_as_failure=True, complementary_rules='',
                          p_bar=True):
        """Class to generate theories for chained positve/negative hard/soft rules.
        Args:
            num (int, optional): number of theories per meta. Defaults to 500.
            chaining_depth (int, optional): Maximum chain length. Defaults to 5.
            negation_as_failure (bool, optional): when set to False, unsatisifed atoms are labeled as 'UNK'.
                                                  Defaults to True.
            complementary_rules (str, optional): a string of rules to include in the LPMLN program,
                                                 ex:   :-spouse(A,A). Defaults to ''.
            p_bar (bool, optional): progress bars. Defaults to True.

        Returns:
            true_theories (List[Dict[str, Any]]): Generated true theories
            false_theories (List[Dict[str, Any]]): Generated false theories
        """

        #  set of symmetric predicates
        with open('data_generation/data/symmetric_preds_set.pkl', 'rb') as f:
            symmetric_preds_set = pickle.load(f)

        rps = [Rule(rule) for rule in self.rules]

        for i, rp in enumerate(rps):
            rp.set_pool_dict(self.pool_lists[i])
            rp.set_rule_support(self.rule_supports[i])

        all_n_pools = []
        true_theories = []
        false_theories = []

        c = 0
        if (p_bar):
            bar = tqdm(total=c)

        max_num_vars = max([rp.num_vars for rp in rps])

        #  find chains in rules
        chains = []
        chain_triples_dict = {}

        # generate chain of depth = 2
        for i, rp in enumerate(rps):
            rp_head = rp.head.relation
            other_rps = [x for x in rps]
            other_rps[i] = None

            for j, other_rp in enumerate(other_rps):
                if(other_rp):
                    other_rp_body_relations = [x.relation for x in other_rp.body]
                    # if rule1 head in rule2 body and rule1 head is not the same as rule2 head, then chain
                    if(rp_head in other_rp_body_relations and rp_head != other_rp.head.relation):
                        chains.append(str(i + 1) + '-' + str(j + 1))
                        chain_triples_dict[str(i + 1) + '-' + str(j + 1)] = \
                            other_rp.body[other_rp_body_relations.index(rp_head)]

        two_chain = chains
        linked_chains = chains

        #  iterate by adding one chain each time until required chain depth is reached
        for i in range(3, chaining_depth + 1, 1):
            ith_chain = linked_chains
            linked_chains = []

            for chain in ith_chain:
                chain_heads = [rps[int(x) - 1].head.relation for x in chain.split('-')]
                for other_chain in two_chain:
                    if(chain.split('-')[-1] == other_chain.split('-')[0]):
                        other_chain_head = rps[int(other_chain.split('-')[-1]) - 1].head.relation
                        if(other_chain_head not in chain_heads):
                            hyphen_ind_chain = [i for i, x in enumerate(chain) if x == '-'][-1]
                            hyphen_ind_other_chain = [i for i, x in enumerate(other_chain) if x == '-'][-1]
                            link = chain.split('-')[-1]
                            linked_chains.append(chain[:hyphen_ind_chain] + '-' + link
                                                 + other_chain[hyphen_ind_other_chain:])
            linked_chains = [x for x in linked_chains if len(set(x.split('-'))) == i]

        while(c < num):

            #  num of additional facts that do are considered as facts
            num_facts = randint(1, 4)
            num_names = randint(max_num_vars, 2 * max_num_vars)

            #  sample pool name
            pool_sample = []
            for rp in rps:
                rp_pool_sample = {}
                for var in rp.vars:
                    #  sample from each pool
                    rp_pool_sample[var] = sample(rp.pool_dict[var], num_names)
                pool_sample.append(rp_pool_sample)

            #  list of all name combinations for each triple in body
            for rp in rps:
                rp_n_pool = []
                for h in rp.body:
                    rp_n_pool.append(list(product(rp.pool_dict[h.subject], rp.pool_dict[h.object])))
                all_n_pools.append(rp_n_pool)

            #  get pool of relations
            rel_pools = []
            for rp in rps:
                rel_pool = [x.relation for x in rp.body]  # get relations from rule body
                rel_pool = list(product(rel_pool, repeat=num_facts))
                br = [x.relation for x in rp.body]
                # heuristic to encourage having predicates of the body of the rule.
                rel_pool = [y for y in rel_pool if sum([y.count(x) for x in set(br)]) >= round(0.8 * num_facts)]
                rel_pools.append(rel_pool)

            # list of all polarity assignments: 3 facts --> 2^3=8 assignments
            assignments = get_assignments(num_facts)[:-1]

            # sample from relation pool
            rels = []
            for i in range(len(rps)):
                rel = sample(rel_pools[i], 1)[0]
                rels.append(rel)

            # sample from assignment pool
            assignmnet = []
            for i in range(len(rps)):
                assignmnet.append(sample(assignments, 1)[0])

            facts = []
            # find facts per assignment
            for i in range(len(rps)):
                for aa, r in zip(assignmnet[i], rels[i]):
                    pool_index = [x.relation for x in rps[i].body].index(r)
                    pp = sample(all_n_pools[i][pool_index], 1)[0]
                    facts.append(generate_fact(pp, aa, r))

            linked_chain = sample(linked_chains, 1)[0]

            ordered_rps = [rps[int(x) - 1] for x in linked_chain.split('-')]

            rules_fired = [r.rule for r in ordered_rps]

            rules_fired_supports = [r.rule_support for r in ordered_rps]

            hyphen_indices = [i for i, x in enumerate(linked_chain) if x == '-']

            chain_triples = []

            for i, x in enumerate(hyphen_indices):
                start = hyphen_indices[i - 1] + 1 if i > 0 else 0
                end = hyphen_indices[i + 1] - 1 if i < (len(hyphen_indices) - 1) else len(linked_chain)
                chain_triples.append(chain_triples_dict[linked_chain[start: end + 1]])

            # generate depth facts
            depth_facts = []
            var_dict_rule = {}

            ordered_pool_sample = []
            for rp in ordered_rps:
                rp_pool_sample = {}
                for var in rp.vars:
                    # sample from each pool
                    rp_pool_sample[var] = sample(rp.pool_dict[var], 7)
                ordered_pool_sample.append(rp_pool_sample)
            pool_sample_dc = deepcopy(ordered_pool_sample)
            hyps_depth = {}

            evidence = []
            for i in range(len(rules_fired) - 1, -1, -1):
                if(var_dict_rule):
                    var_switch_dict = {'A': chain_triples[i].subject, 'B': chain_triples[i].object}
                    var_dict_rule = {k: var_dict_rule[var_switch_dict[k]] for (k, v) in var_dict_rule.items()
                                     if k in var_switch_dict}

                pool_sample_dc[i].update(var_dict_rule)

                facts_rule, var_dict_rule = ordered_rps[i].random_sample_with_assignment(pool_sample_dc[i])
                self.facts_rule = facts_rule
                self.var_dict_rule = var_dict_rule
                self.pool_sample_dc = pool_sample_dc[i]
                evidence.append('(' + ordered_rps[i].rule + ',' + " ".join(facts_rule) + ')')
                # remove chained triple
                if(i > 0):
                    pred2remove = [x for x in facts_rule if chain_triples[i - 1].relation in x][0]
                    facts_rule = [x for x in facts_rule if x != pred2remove]

                var_dict_rule = {k: [v] for (k, v) in var_dict_rule.items()}

                hyps_depth[str(i + 1)] = ordered_rps[i].head.ground(var_dict_rule['A'][0], var_dict_rule['B'][0])
                depth_facts.append(facts_rule)

            depth_facts = [item for list_ in depth_facts for item in list_]
            evid = "\t ==>\n".join(evidence[::-1])

            all_generated_facts = [x for x in facts if x not in depth_facts]

            facts = sample(all_generated_facts, num_facts) if len(all_generated_facts) > num_facts \
                else all_generated_facts

            random_facts = [x for x in all_generated_facts if x not in facts] if facts != all_generated_facts else []

            facts_aug_split = list(set(facts + depth_facts))
            facts_aug = " ".join(facts_aug_split)

            #  solve
            if(self.lpmln):
                lpmln_prob = solve_LPMLN_multi(facts_aug, rules_fired, rules_fired_supports, complementary_rules, 5)
                if(not lpmln_prob):
                    continue

            asp_answer_aug = solve_ASP(facts_aug, " ".join(rules_fired))
            self.lpmln_prob = lpmln_prob
            for depth in hyps_depth:
                rconc = hyps_depth[depth]
                p_theory = {'facts': facts_aug,
                            'hypothesis': rconc,
                            'rule': rules_fired,
                            'output': True,
                            'solution': asp_answer_aug,
                            'meta': f'rconc_depth_{depth}',
                            'evidence': evid}
                h = get_hash(p_theory['facts'] + " ".join(p_theory['rule']) + p_theory['hypothesis'] + p_theory['meta'])
                p_theory['id'] = f'Chain_{h}'
                if(self.lpmln):
                    p_theory['hyp_weight'] = get_proba_lpmln(lpmln_prob, rconc)
                    p_theory['rule_support'] = rules_fired_supports
                    p_theory['solution'] = lpmln_prob

                true_theories.append(symbolic2text(p_theory))

                # check if non-symmetric and switch 50% of the time
                if(Triple(rconc).relation not in symmetric_preds_set
                    and negate(Triple(rconc).relation) not in symmetric_preds_set
                        and random() > 0.5):

                    f_theory = {'facts': facts_aug,
                                'hypothesis': switch(rconc),
                                'rule': rules_fired,
                                'output': False,
                                'solution': asp_answer_aug,
                                'meta': f'switch_rconc_depth_{depth}'}
                    h = get_hash(f_theory['facts'] + " ".join(f_theory['rule']) + f_theory['hypothesis']
                                 + f_theory['meta'])
                    f_theory['id'] = f'Chain_{h}'
                    if(self.lpmln):
                        f_theory['hyp_weight'] = get_proba_lpmln(lpmln_prob, switch(rconc))
                        f_theory['rule_support'] = rules_fired_supports
                        f_theory['solution'] = lpmln_prob

                    false_theories.append(symbolic2text(f_theory))

                else:
                    # negate satisfied positive rule head
                    f_theory = {'facts': facts_aug,
                                'hypothesis': negate(rconc),
                                'rule': rules_fired,
                                'output': False,
                                'solution': asp_answer_aug,
                                'meta': f'inv_rconc_depth_{depth}',
                                'evidence': evid}
                    h = get_hash(f_theory['facts'] + " ".join(f_theory['rule']) + f_theory['hypothesis']
                                 + f_theory['meta'])
                    f_theory['id'] = f'Chain_{h}'
                    if(self.lpmln):
                        f_theory['hyp_weight'] = get_proba_lpmln(lpmln_prob, negate(rconc))
                        f_theory['rule_support'] = rules_fired_supports
                        f_theory['solution'] = lpmln_prob

                    false_theories.append(symbolic2text(f_theory))

            if(random_facts):
                pos_random_facts = [x for x in random_facts if x[:3] != 'neg']

                unsat_fact_atom = sample(pos_random_facts, 1)[0][:-1]

                f_theory = {'facts': facts_aug,
                            'hypothesis': unsat_fact_atom,
                            'rule': rules_fired,
                            'output': False if negation_as_failure else 'UNK',
                            'solution': asp_answer_aug,
                            'meta': 'unsat_fact'}
                h = get_hash(f_theory['facts'] + " ".join(f_theory['rule']) + f_theory['hypothesis']
                             + f_theory['meta'])
                f_theory['id'] = f'Chain_{h}'

                if(self.lpmln):
                    f_theory['hyp_weight'] = get_proba_lpmln(lpmln_prob, unsat_fact_atom)
                    f_theory['rule_support'] = rules_fired_supports
                    f_theory['solution'] = lpmln_prob

                false_theories.append(symbolic2text(f_theory))

                p_theory = {'facts': facts_aug,
                            'hypothesis': negate(unsat_fact_atom),
                            'rule': rules_fired,
                            'output': True if negation_as_failure else 'UNK',
                            'solution': asp_answer_aug,
                            'meta': 'inv_unsat_fact'}
                h = get_hash(p_theory['facts'] + " ".join(p_theory['rule']) + p_theory['hypothesis'] + p_theory['meta'])
                p_theory['id'] = f'Chain_{h}'
                if(self.lpmln):
                    p_theory['hyp_weight'] = get_proba_lpmln(lpmln_prob, negate(unsat_fact_atom))
                    p_theory['rule_support'] = rules_fired_supports
                    p_theory['solution'] = lpmln_prob

                true_theories.append(symbolic2text(p_theory))

            # sample positive fact
            if(facts):
                sat_fact = sample(facts, 1)[0][:-1]
                sat_fact_meta = 'depth_0'

                # if symmetric fact then switch subj and obj 50% of the time
                if((Triple(sat_fact).relation in symmetric_preds_set
                    or negate(Triple(sat_fact).relation) in symmetric_preds_set)
                        and random() > 0.5):

                    sat_fact = switch(sat_fact)
                    sat_fact_meta = 'switch_depth_0'

                p_theory = {'facts': facts_aug,
                            'hypothesis': sat_fact,
                            'rule': rules_fired,
                            'output': True,
                            'solution': asp_answer_aug,
                            'meta': sat_fact_meta}
                h = get_hash(p_theory['facts'] + " ".join(p_theory['rule']) + p_theory['hypothesis'] + p_theory['meta'])
                p_theory['id'] = f'Chain_{h}'
                if(self.lpmln):
                    p_theory['hyp_weight'] = get_proba_lpmln(lpmln_prob, sat_fact)
                    p_theory['rule_support'] = rules_fired_supports
                    p_theory['solution'] = lpmln_prob

                true_theories.append(symbolic2text(p_theory))

                # if not symmetric then switch
                if(Triple(sat_fact).relation not in symmetric_preds_set
                    and negate(Triple(sat_fact).relation) not in symmetric_preds_set
                        and random() > 0.5):
                    f_theory = {'facts': facts_aug,
                                'hypothesis': switch(sat_fact),
                                'rule': rules_fired,
                                'output': False,
                                'solution': asp_answer_aug,
                                'meta': 'switch_fact'}
                    h = get_hash(f_theory['facts'] + " ".join(f_theory['rule']) + f_theory['hypothesis']
                                 + f_theory['meta'])
                    f_theory['id'] = f'Chain_{h}'
                    if(self.lpmln):
                        f_theory['hyp_weight'] = get_proba_lpmln(lpmln_prob, switch(sat_fact))
                        f_theory['rule_support'] = rules_fired_supports
                        f_theory['solution'] = lpmln_prob

                    false_theories.append(symbolic2text(f_theory))

                else:
                    f_theory = {'facts': facts_aug,
                                'hypothesis': negate(sat_fact),
                                'rule': rules_fired,
                                'output': False,
                                'solution': asp_answer_aug,
                                'meta': 'inv_fact'}
                    h = get_hash(f_theory['facts'] + " ".join(f_theory['rule']) + f_theory['hypothesis']
                                 + f_theory['meta'])
                    f_theory['id'] = f'Chain_{h}'

                    if(self.lpmln):
                        f_theory['hyp_weight'] = get_proba_lpmln(lpmln_prob, negate(sat_fact))
                        f_theory['rule_support'] = rules_fired_supports
                        f_theory['solution'] = lpmln_prob

                    false_theories.append(symbolic2text(f_theory))

            c += 1

            # if(c % 10):
            #     json.dump(true_theories, open(f"true_theories_chain_{chaining_depth}.json",'w'))
            #     json.dump(false_theories, open(f"false_theories_chain_{chaining_depth}.json",'w'))

            if (p_bar):
                bar.update(1)

        return true_theories, false_theories


def main():

    my_parser = argparse.ArgumentParser(description='Generate theories from input rules.')

    my_parser.add_argument('--rule_json',
                           type=str,
                           help='dict containing rule, pool lists, and rule support')

    my_parser.add_argument('--rule',
                           type=str,
                           help='rule')

    my_parser.add_argument('--pool_list',
                           type=str,
                           help='pool list')

    my_parser.add_argument('--type',
                           type=str,
                           default='single',
                           choices=['single', 'union', 'chain'],
                           help='single/union/chain')

    my_parser.add_argument('--chain_depth',
                           type=int,
                           default=5,
                           help='Chaining Depth')

    my_parser.add_argument('--rule_support',
                           type=float,
                           default=None,
                           help='Rule Support')

    my_parser.add_argument('--max_num_facts',
                           type=int,
                           default=7,
                           help='maximum number of facts in a theory')

    my_parser.add_argument('--num',
                           type=int,
                           default=20,
                           help='Total number of theories per (rule,facts).')

    my_parser.add_argument('--train_size',
                           type=float,
                           default=0.7,
                           help='Proportion of the dataset to be included in the training set.')

    my_parser.add_argument('--val_size',
                           type=float,
                           default=0.1,
                           help='Proportion of the dataset to be included in the validation set. \
                                 The rest will be included in the test set.')

    my_parser.add_argument('--TWL',
                           action='store_true',
                           default=False,
                           help='Three-way logic: output UNK for unsatisfied predicates')

    my_parser.add_argument('--complementary_rules',
                           type=str,
                           default='',
                           help='Complementary rules added to the program')

    my_parser.add_argument('--p_bar',
                           type=bool,
                           default=True,
                           help='Show progress bar')

    args = my_parser.parse_args()

    if (not args.rule and not args.rule_json):
        print("Please specify data for the rule.")
        exit()

    # if a json file is included in the input, we load it and update the parameters
    if args.rule_json:
        rule_json = json.load(open(args.rule_json))
        args = vars(args)
        args.update(rule_json)
        args = argparse.Namespace(**args)

    # compatability for the case when strings are inputted
    if type(args.rule) == str:
        args.rule = [args.rule]
        args.pool_list = [ast.literal_eval(args.pool_list)] if isinstance(args.pool_list, str) else [args.pool_list]
        args.rule_support = [args.rule_support]
    print(args)

    if args.type == 'single':

        if 'rules' in args:
            args.rule = args.rules

        if 'pool_lists' in args:
            args.pool_list = args.pool_lists

        if 'rule_supports' in args:
            args.rule_support = args.rule_supports

        for i in range(len(args.rule)):
            dg = OneRuleDataGenerator(args.rule[i],
                                      args.pool_list[i],
                                      args.max_num_facts,
                                      args.rule_support[i])
            true_theories, false_theories = dg.generate_theories(num=args.num,
                                                                 negation_as_failure=not args.TWL,
                                                                 complementary_rules=args.complementary_rules,
                                                                 p_bar=args.p_bar)

            folder_name = args.rule[i].replace(" ", "") if args.rule[i] else args.rule_json.split("json")[0][0:-1]
            path = f'data/generated/{args.type}/{folder_name}_{time.strftime("%Y%m%dT%H%M%S")}/'
            Path(path).mkdir(parents=True, exist_ok=True)
            combined = true_theories + false_theories
            train,val_test = train_test_split(combined, train_size=args.train_size)
            val,test = train_test_split(val_test, train_size=int(args.val_size * len(combined)))
            dump_jsonl(train, "train.jsonl", path)
            dump_jsonl(val, "val.jsonl", path)
            dump_jsonl(test, "test.jsonl", path)

    # We could have passed the args dict to the function and ignored unwanted keys by adding **kwargs
    # to the function definition, but I prefer to keep it this way for readibility.
    else:
        if args.type == 'union':
            dg = MultiRuleDataGenerator(args.rules,
                                        args.pool_lists,
                                        args.max_num_facts,
                                        args.rule_supports)

            true_theories, false_theories = dg.generate_theories(args.num)

        elif args.type == 'chain':
            dg = ChainedRulesDataGenerator(args.rules,
                                           args.pool_lists,
                                           args.rule_supports)

            true_theories, false_theories = dg.generate_theories(args.num,
                                                                 args.chain_depth)

        folder_name = args.rule.replace(" ", "") if args.rule else args.rule_json.split("json")[0][0:-1]
        path = f'data/generated/{args.type}/{folder_name}_{time.strftime("%Y%m%dT%H%M%S")}/'
        Path(path).mkdir(parents=True, exist_ok=True)
        combined = true_theories + false_theories
        train,val_test = train_test_split(combined, train_size=args.train_size)
        val,test = train_test_split(val_test, train_size=int(args.val_size * len(combined)))
        dump_jsonl(train, "train.jsonl", path)
        dump_jsonl(val, "val.jsonl", path)
        dump_jsonl(test, "test.jsonl", path)


if(__name__ == "__main__"):
    main()
