from . import util
from itertools import chain
import clingo

class XOR:
    """
    A XOR constraint maintains the following invariants:
    1. there are at least two literals, and
    2. the first two literals are unassigned, or all literals are assigned and
       the first two literals have been assigned last on the same decision
       level.
    Furthermore, an index pointing to the literal after the literal assigned
    last is maintained. We start the search for the next unassigned literal
    from this point. This is important to get the amortized linear propagation
    time.
    """
    def __init__(self, literals):
        assert(len(literals) >= 2)
        self.__literals = literals
        self.__index = 2

    def __len__(self):
        return len(self.__literals)

    def __getitem__(self, idx):
        return self.__literals[idx]

    def __setitem__(self, idx, val):
        self.__literals[idx] = val
        return val

    def propagate(self, assignment, i):
        """
        Propagates the given assigned index.

        If an unwatched unassigned literal is found, the literals are
        rearranged so that the given index points to it. The function returns
        true if an such a literal is found.
        """
        assert(i < 2)
        for j in chain(range(self.__index, len(self)), range(2, self.__index)):
            if assignment.value(self[j]) is None:
                self.__index = j + 1 if j + 1 < len(self) else 2
                self[i], self[j] = self[j], self[i]
                return True
        return False

    def reason(self, assignment, i):
        """
        If the constraint is unit resulting or conflicting returns a reason in
        form of a clause.
        """
        # Switch to the index of the other watched literal that is either
        # unassigned and has to be propagated or has to be checked for a
        # conflict. In the second case it was assigned on the same level as the
        # propagated literal.
        i = 1 - i
        count = 0
        clause = []
        for j in range(len(self)):
            if i == j:
                continue
            if assignment.is_true(self[j]):
                clause.append(-self[j])
                count += 1
            else:
                clause.append(self[j])

        clause.append(-self[i] if count % 2 else self[i])

        return None if assignment.is_true(clause[-1]) else clause

class WatchesUnitPropagator:
    def __init__(self):
        self.__states  = []
        self.__sat = True
        self.__consequences = []

    def __add_watch(self, ctl, xor, unassigned, thread_ids):
        """
        Adds a watch for the for the given index.

        The literal at the given index has to be either unassigned or become
        unassigned through backtracking before the associated constraint can
        become unit resulting again.
        """
        variable = abs(xor[unassigned])
        ctl.add_watch( variable)
        ctl.add_watch(-variable)
        for thread_id in thread_ids:
            self.__states[thread_id].setdefault(variable, []).append((xor, unassigned))

    def init(self, init):
        """
        Initializes xor_1 constraints and watches based on the symbol table.

        Constraints of length zero and one are handled specially, to keep the
        implementation of the general constraints simple.
        """
        for thread_id in range(len(self.__states), init.number_of_threads):
            self.__states.append({})

        ret = util.symbols_to_xor_r(init.symbolic_atoms, util.default_get_lit(init))
        if ret is None:
            self.__sat = False
        else:
            constraints, facts = ret
            self.__consequences.extend(facts)
            for constraint in constraints:
                xor = XOR(constraint)
                self.__add_watch(init, xor, 0, range(init.number_of_threads))
                self.__add_watch(init, xor, 1, range(init.number_of_threads))

        init.check_mode = clingo.PropagatorCheckMode.Fixpoint

    def check(self, control):
        """
        Since the XOR constraint above handles only constraints with at least
        two literals, here the other two cases are handled.

        Empty conflicting constraints result in top-level conflicts and unit
        constraints will be propagated on the top-level.
        """
        if control.assignment.decision_level == 0:
            if not self.__sat:
                control.add_clause([]) and control.propagate()
                return
            for lit in self.__consequences:
                if not control.add_clause([lit]) or not control.propagate():
                    return

    def propagate(self, control, changes):
        """
        Propagates XOR constraints maintaining two watches per constraint.

        Generated conflicts are guaranteed to be asserting (have at least two
        literals from the current decision level).
        """
        state  = self.__states[control.thread_id]
        for literal in changes:
            variable = abs(literal)

            state[variable], watches = [], state[variable]
            assert(len(watches) > 0)
            for i in range(len(watches)):
                xor, unassigned = watches[i]
                if xor.propagate(control.assignment, unassigned):
                    # We found an unassigned literal, which is watched next.
                    self.__add_watch(control, xor, unassigned, (control.thread_id,))
                else:
                    # Here the constraint is either unit, satisfied, or
                    # conflicting. In any case, we can keep the watch because
                    # (*) the current decision level has to be backtracked
                    # before the constraint can become unit again.
                    state[variable].append((xor, unassigned))

                    clause = xor.reason(control.assignment, unassigned)
                    if clause is not None:
                        if not control.add_clause(clause) or not control.propagate():
                            assert(state[variable])
                            # reestablish the remaining watches with the same
                            # reason as in (*)
                            state[variable].extend(watches[i + 1:])
                            return

            if len(state[variable]) == 0:
                control.remove_watch( variable)
                control.remove_watch(-variable)
                state.pop(variable)
