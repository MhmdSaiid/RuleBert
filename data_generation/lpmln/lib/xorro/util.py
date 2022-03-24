from collections import namedtuple
from itertools import *
from functools import *
import sys
from textwrap import dedent as _dedent
from math import log
from random import randint, sample


def attrdef(m, a, b):
    return getattr(m, a if hasattr(m, a) else b)

zip_longest = attrdef(sys.modules[__name__], 'zip_longest', 'izip_longest')
reduce = getattr(sys.modules[__name__], 'reduce')



def get_parity(par):
    if str(par) == "odd": return 1
    else: return 0

def invert_parity(par):
    return int(par) ^ 1

def get_str_parity(par):
    if par == 1: return "odd"
    else: return "even"

def default_get_lit(init):
    value = init.assignment.value
    def get_lit(atom):
        lit = init.solver_literal(atom.literal)
        return lit, value(lit)
    return get_lit

class _XORConstraint:
    def __init__(self, parity):
        self.parity = parity
        self.literals = set()

def symbols_to_xor_r(symbolic_atoms, get_lit):
    """
    Returns None if the constraints are trivially unsatisfiable, otherwise
    returns a list of xor_1 constraints and a list of facts. A xor_1 constraint is
    represented as a list of literals.

    Arguments:
    symbolic_atoms -- The domain having predicates __parity/2 and __parity/3.
    get_lit        -- Function mapping a symbolic atom to a litral and its
                      truth value.
    """
    constraints = {}
    lits = []
    for atom in symbolic_atoms.by_signature("__parity",2):
        cid = atom.symbol.arguments[0].number
        par = atom.symbol.arguments[1].name
        constraints[cid] = _XORConstraint(get_parity(par))

    for atom in symbolic_atoms.by_signature("__parity",3):
        constraint = constraints[atom.symbol.arguments[0].number]
        lit, truth = get_lit(atom)

        if truth:
            constraint.parity = invert_parity(constraint.parity)
        elif truth is None:
            if lit in constraint.literals:
                constraint.literals.remove(lit)
            elif -lit in constraint.literals:
                constraint.literals.remove(-lit)
                constraint.parity = invert_parity(constraint.parity)
            else:
                if lit < 0:
                    constraint.literals.add(abs(lit))
                    constraint.parity = invert_parity(constraint.parity)
                else:
                    constraint.literals.add(lit)
                if abs(lit) not in lits:
                    lits.append(abs(lit))
                
    facts = set()
    result = []
    for constraint in constraints.values():
        literals = sorted(constraint.literals)
        n = len(literals)
        if n == 0:
            if constraint.parity == 1:
                return None
        else:
            if constraint.parity == 0:
                literals[0] = -literals[0]
            if n > 1:
                result.append(literals)
            else:
                facts.add(literals[0])

    return result, sorted(facts)


def generate_random_xors(prg, files, s, q,deduct=0):
    """
    Of course adding the theory may not be the best way to do it. This is just a hack
    Maybe using the AST is a better alternative to extract the symbols to build the xor_1 constraints.
    In the end, we just need the symbols to write a file with random constraints. 
    """
    '''for f in files:
        prg.load(f)'''

    prg.add("base",[],files)

    prg.add("base", [], _dedent("""\
      #theory parity { 
        element {}; 
        &odd/0 : element, directive; 
        &even/0 : element, directive }.
      """))

    prg.ground([("base", [])])

    estimated_s = s
    xors  = ""
    symbols = [atom.symbol for atom in prg.symbolic_atoms if atom.is_fact is False and "__parity" not in str(atom.symbol)]


    if len(symbols) > 0:
        if s == 0:
            estimated_s = int(log(len(symbols) + 1, 2))
            estimated_s-=deduct
        for i in range(estimated_s):
            range_ = int((len(symbols))*q)
            size = randint(range_, range_)
            terms = " ; ".join(str(x)+":"+str(x) for x in sorted(sample(symbols, size)))
            xors  += "&%s{ %s }.\n"%(get_str_parity(randint(0,1)), terms)

    return xors
