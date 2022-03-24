"""
This xor_1 propagator actually does not interfere with clasp's propagation.  In
fact we just check (count) the number of truth's assignments given by clasp
respecting the parity given for each xor_1 constraint In case of conflict, add
the nogood and let clasp to propagate again
"""

from . import util

class XOR:
    def __init__(self, literals):
        self.__literals = literals

    def check(self, ass):
        if not sum(1 for lit in self if ass.is_true(lit)) % 2:
            return [lit if ass.is_true(lit) else -lit for lit in self]

    def __iter__(self):
        return iter(self.__literals)

class CountCheckPropagator:
    def __init__(self):
        self.__states = []

    def init(self, init):
        for thread_id in range(len(self.__states), init.number_of_threads):
            self.__states.append([])

        # NOTE: quite a bit of ceremony here
        #       it would be better to handle these cases here more elegantly
        #       but this propagator is just a toy anyway
        ret = util.symbols_to_xor_r(init.symbolic_atoms, util.default_get_lit(init))
        if ret is None:
            constraints = [[]]
        else:
            constraints, facts = ret
            constraints.extend([fact] for fact in facts)

        for constraint in constraints:
            xor = XOR(list(sorted(constraint)))
            self.__states[thread_id].append(xor)

    def check(self, control):
        state = self.__states[control.thread_id]
        for xor in state:
            nogood = xor.check(control.assignment)
            if nogood is not None:
                control.add_nogood(nogood) and control.propagate()
                return
