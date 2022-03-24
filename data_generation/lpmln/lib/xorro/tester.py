"""
The xorro module contains functions to solve logic programs with parity
constraints.

Classes:
Application -- Main application class.

Functions:
main  -- Main function starting an extended clingo application.
"""

from . import util
from . import transformer as _tf
from .countp import CountCheckPropagator
from .watches_up import WatchesUnitPropagator
from .propagate_gje import Propagate_GJE
from .reason_gje import Reason_GJE
from random import sample
import sys as _sys
import clingo as _clingo
from textwrap import dedent as _dedent


import clingo


def translate_binary_xor(backend, lhs, rhs):
    aux = backend.add_atom()
    backend.add_rule([aux], [lhs, -rhs])
    backend.add_rule([aux], [-lhs, rhs])
    return aux



def transform(prg, files, xors=""):
    with prg.builder() as b:
        _tf.transform(files, b.add, xors)


class Leaf:
    def __init__(self, atom):
        self.__atom = atom

    def translate(self, backend):
        return self.__atom


class Tree:
    def __init__(self, lhs, rhs):
        self.__lhs = lhs
        self.__rhs = rhs

    def translate(self, backend):
        lhs = self.__lhs.translate(backend)
        rhs = self.__rhs.translate(backend)
        return translate_binary_xor(backend, lhs, rhs)


class List:
    def __init__(self, literals):
        assert (len(literals) > 0)
        self.__literals = literals

    def translate(self, backend):
        return util.reduce(lambda l, r: translate_binary_xor(backend, l, r), self.__literals)


def translate(mode, prg, cutoff):
    if mode == "count":
        prg.add("__count", [], _dedent("""\
            :- { __parity(ID,even,X) } = N, N\\2!=0, __parity(ID,even).
            :- { __parity(ID,odd ,X) } = N, N\\2!=1, __parity(ID,odd).
            """))
        prg.ground([("__count", [])])

    elif mode == "countp":
        prg.register_propagator(CountCheckPropagator())

    elif mode == "up":
        prg.register_propagator(WatchesUnitPropagator())

    elif mode == "gje":
        prg.register_propagator(WatchesUnitPropagator())
        prg.register_propagator(Propagate_GJE(cutoff))

    elif mode == "reason-gje":
        prg.register_propagator(Reason_GJE(cutoff))

    elif mode in ["list", "tree"]:
        def to_tree(constraint):
            layer = [Leaf(literal) for literal in constraint]

            def tree(l, r):
                return l if r is None else Tree(l, r)

            while len(layer) > 1:
                layer = list(util.starmap(tree, util.zip_longest(layer[0::2], layer[1::2])))
            return layer[0]

        def get_lit(atom):
            return atom.literal, True if atom.is_fact else None

        ret = util.symbols_to_xor_r(prg.symbolic_atoms, get_lit)
        with prg.backend() as b:
            if ret is None:
                b.add_rule([], [])
            else:
                constraints, facts = ret
                for fact in facts:
                    b.add_rule([], [-fact])
                for constraint in constraints:
                    tree = List(constraint) if mode == "list" else to_tree(constraint)
                    b.add_rule([], [-tree.translate(b)])

    else:
        raise RuntimeError("unknow transformation mode: {}".format(mode))


class Application:
    """
    Application object as accepted by clingo.clingo_main().

    Rewrites the parity constraints in logic programs into normal ASP programs
    and solves them.
    """

    def __init__(self,args):
        """
        Initializes the application setting the program name.

        See clingo.clingo_main().
        """
        self.version = "1.0"
        self.__approach = "count"
        self.__cutoff = 0.0
        self.__s = 0
        self.__q = 0.2
        self.args = args

    def __parse_approach(self, value):
        """
        Parse approach argument.
        """
        self.__approach = str(value)
        return self.__approach in ["count", "list", "tree", "countp", "up", "gje", "reason-gje"]

    def __parse_cutoff(self, value):
        """
        Parse cutoff argument.
        """
        self.__cutoff = float(value)
        return self.__cutoff >= 0.0 and self.__cutoff <= 1.0

    def __parse_s(self, value):
        """
        Parse s value as the number of xor_1 constraints.
        """
        self.__s = int(value)
        return self.__s >= 0

    def __parse_q(self, value):
        """
        Parse the q argument for random xor_1 constraints.
        """
        self.__q = float(value)
        return self.__q >= 0.0 and self.__q <= 1.0

    def register_options(self, options):
        """
        Extension point to add options to xorro like choosing the
        transformation to apply.

        """
        group = "Xorro Options"
        options.add(group, "approach", _dedent("""\
        Approach to handle XOR constraints [count]
              <arg>: {count|list|tree|countp|up|gje}
                count      : Add count aggregates modulo 2
                {list,tree}: Translate binary xor_1 operators to rules
                             (binary operators are arranged in list/tree)
                countp     : Propagator simply counting assigned literals
                up         : Propagator implementing unit propagation
                gje        : Propagator implementing Gauss-Jordan Elimination"""), self.__parse_approach)

        options.add(group, "cutoff", _dedent("""\
        Cutoff percentage of literals assigned before GJE [0-1]
                """), self.__parse_cutoff)

        options.add_flag(group, "sampling", _dedent("""\
        Enable sampling by generating random xor_1 constraints"""), self.__sampling)

        options.add(group, "s", _dedent("""\
        Number of xor_1 constraints to generate. Default=0, log(#atoms)"""), self.__parse_s)

        options.add(group, "q", _dedent("""\
        Density of each xor_1 constraint. Default=0.5"""), self.__parse_q)

        options.add_flag(group, "display", _dedent("""\
        Display the random xor_1 constraints used in sampling"""), self.__display)

    def main(self, prg, files, deduct):

        s = self.__s
        q = self.__q
        xors = util.generate_random_xors(prg, files, s, q, deduct)
        transform(prg, files, xors)
        prg.ground([("base", [])])
        translate(self.__approach, prg, self.__cutoff)

        prg.configuration.solve.models = 0
        models = []
        selected = []
        requested_models = int(str(prg.configuration.solve.models))

        ret = prg.solve(None, lambda model: models.append(model.symbols(atoms=True)))

        if self.args.verbosity>6:
            print("# of models returned: ", len(models))

        if requested_models == -1:
            requested_models = 1
        elif requested_models == 0:
            requested_models = len(models)
        if str(ret) == "SAT":
            if requested_models > len(models):
                requested_models = len(models)
            selected = sorted(sample(range(1, len(models) + 1), requested_models))
            md4return = []
            for i in range(requested_models):
                md4return.append(models[selected[i] - 1])
            return md4return
