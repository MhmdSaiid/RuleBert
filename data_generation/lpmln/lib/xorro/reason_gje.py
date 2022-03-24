from . import util
from . import gje
from itertools import chain
import clingo

def xor_columns(col, parity):
    result = []
    for i in range(len(col)):
        result.append(col[i] ^ parity[i])
    return result

def lits_to_binary(columns, lits, constraint):
    for i in range(len(lits)):
        if lits[i] in constraint:
            columns.setdefault(lits[i], []).append(1)
        elif -lits[i] in constraint:
            columns.setdefault(lits[i], []).append(1)
        else:
            columns.setdefault(lits[i], []).append(0)
    return columns

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

    def reason_gje(self, columns, assignment, n_lits, cutoff):
        state = {}
        partial = []
        deduced_literals = []

        ## Get Partial Assignment
        state["parity"] = columns["parity"]
        for lit, column in columns.items():
            if lit != "parity":
                value = assignment.value(lit)
                if value == None:
                    state[lit] = column
                elif value == True:
                    state["parity"] = xor_columns(column, state["parity"])
                    partial.append( lit)
                elif value == False:                    
                    partial.append(-lit)                           

        ## Build the matrix from columns state
        matrix, xor_lits= gje.columns_state_to_matrix(state)

        ## Percentage of assigned literals
        assigned_lits_perc = 1.0-float("%.1f"%(len(xor_lits)/n_lits))
        ## If there are more than unary xors perform GJE
        if len(state) > 2 and assigned_lits_perc >= cutoff:
            matrix = gje.remove_rows_zeros(matrix)
            matrix = gje.perform_gauss_jordan_elimination(matrix, False)

        ## Check SATISFIABILITY
        conflict = gje.check_sat(matrix)
        if not conflict and xor_lits:
            ## Imply literals 
            deduced_literals = gje.deduce_clause(matrix, xor_lits)

        return conflict, partial, deduced_literals


class Reason_GJE:
    def __init__(self, cutoff):
        self.__states  = []
        self.__columns = []
        self.__sat = True
        self.__consequences = []
        self.__cutoff = cutoff
        self.__n_literals = 0

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
        Initializes xor_1 constraints based on the symbol table to build a binary matrix.
        This propagator is called on fixpoints to perform Gauss-Jordan Elimination after Unit Propagation
        """
        for thread_id in range(len(self.__states), init.number_of_threads):
            self.__states.append({})
            self.__columns.append({})

        init.check_mode = clingo.PropagatorCheckMode.Fixpoint
        literals = []
        for atom in init.symbolic_atoms.by_signature("__parity",3):
            lit = init.solver_literal(atom.literal)
            value = init.assignment.value(lit)
            if value == None and abs(lit) not in literals:
                literals.append(abs(lit))
        
        ## Get the constraints
        ret = util.symbols_to_xor_r(init.symbolic_atoms, util.default_get_lit(init))
        # Number of literals in GJ Matrix
        self.__n_literals = float(len(literals))
        
        if ret is None:
            self.__sat = False
        elif ret is not None:
            # NOTE: whether facts should be handled here is up to question
            #       this should only be necessary if the propagator is to be used standalone
            #       without any of the other approaches
            constraints, facts = ret
            self.__consequences.extend(facts)

            ## Get the literals and parities
            for constraint in constraints:
                # FIXME: check if there is another way to do this. All constraints are represented as "odd" constraints but GJE only uses non-negative variables/literals.
                # Somehow we need to convert xor_1 constraints with a negative into a positive literal and invert the parity to build the matrix.
                even = False
                if constraint[0] < 0:
                    even = True
                for thread_id in range(init.number_of_threads):
                    if even:
                        self.__columns[thread_id].setdefault("parity", []).append(0)
                    else:
                        self.__columns[thread_id].setdefault("parity", []).append(1)

            ## Build the rest of the matrix
            for constraint in constraints:
                if len(constraint) == 1:
                    lit = next(iter(constraint))
                    self.__consequences.append(lit if constraint[0] > 1 else -lit)
                elif len(constraint):
                    xor = XOR(constraint)
                    self.__add_watch(init, xor, 0, range(init.number_of_threads))
                    self.__add_watch(init, xor, 1, range(init.number_of_threads))
                
                for thread_id in range(init.number_of_threads):
                    self.__columns[thread_id] = lits_to_binary(self.__columns[thread_id], sorted(literals), constraint)
                    
        else:
            # NOTE: if the propagator is to be used standalone, this case has to be handled
            pass

    def check(self, control):
        """
        Check if current assignment is conflict-free, detect a conflict or deduce literals
        by doing Gauss-Jordan Elimination
        """
        """
        Since the XOR constraint above handles only constraints with at least
        two literals, here the other two cases are handled.

        Empty conflicting constraints result in top-level conflicts and unit
        constraints will be propagated on the top-level.
        """
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
        state = self.__states[control.thread_id]
        columns = self.__columns[control.thread_id]
        n = self.__n_literals
        cutoff = self.__cutoff
        
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
                    
                    ## GJE
                    conflict, partial, clause = xor.reason_gje(columns, control.assignment, n, cutoff)
                    if clause is not None:
                        for lit in clause:
                            if not control.add_nogood(partial+[-lit]) or not control.propagate():
                                return                                
                    if conflict:
                        if not control.add_nogood(partial) or not control.propagate():
                            return
                    
            if len(state[variable]) == 0:
                control.remove_watch( variable)
                control.remove_watch(-variable)
                state.pop(variable)
