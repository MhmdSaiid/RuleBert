try:
    from lib import sdd as sddm
except Exception:
    raise Exception("SDD library is not yet installed. To install, please invoke lp2sdd with the option --install")
    

from src import logic
from collections import *
import signal



class TwoValuedCompiledProgram:
    """
    A representation of a two-valued compiled logic program: for each atom, an SDD representing its value in the
    WFM. Furthermore, self.constraint contains an SDD compilation of the constraints attached to this program.
    In case of Stable Semantics, self.constraints also contains the condition that every atom corresponds to the result
    of applying the GL reduct and computing bottom-up.

    This result also has the original program stored in self.program

    """

    def __init__(self, atom2sdd, atom2sddnum, manager, program):
        self.atom2sdd = atom2sdd
        self.manager = manager
        self.constraint = sddm.sdd_manager_true(self.manager)
        self._entire_program_as_sdd = None
        # NOTE: in atom2sddnum, it is possible that multiple atoms map to the same number:
        # This happens when equivalences between two variables are detected, and at least one of them is internal
        self.atom2sddnum = atom2sddnum
        self.program = program

    def addconstraint(self, formula):
        """
        Adds a constraint to this data structure. Conjoins self.constraint with the compilation of the formula
        :param formula: the formula to compile. Literals in this constraint are replaced by their value according to
        self.get_literal_value
        :return: nothing
        """
        new_constraint = compile_formula(formula, self.manager, self.get_literal_value)
        old_constraint = self.constraint
        self.constraint = sdd_conjoin([old_constraint, new_constraint], self.manager)
        sddm.sdd_deref(new_constraint, self.manager)
        sddm.sdd_deref(old_constraint, self.manager)

    def addconstraint_without_evaluating(self, formula):
        """
        Adds a constraint to this data structure. Conjoins self.constraint with the compilation of the formula
        :param formula: the formula to compile. Literals in this constraint are replaced by the sdd var number stored
        for them in self.atom2sdd if existing, and to a new sdd variable otherwise.
        :return: nothing
        """
        new_constraint = compile_formula(formula, self.manager, self.get_literal_sdd_leaf)
        old_constraint = self.constraint
        self.constraint = sdd_conjoin([old_constraint, new_constraint], self.manager)
        sddm.sdd_deref(new_constraint, self.manager)
        sddm.sdd_deref(old_constraint, self.manager)

    def get_literal_sdd_leaf(self, lit):
        """
        Returns an SDD representing the literal value.
        In case of a negated literal, returns the negation of the sdd representing the underlying atom
        :param lit: the literal to represent
        :return: An sdd representing this literal. Increases the reference count of this sdd by one
        """
        negated = lit.negated
        if negated:
            lit = lit.negate()
        num = self.get_sdd_var_num(lit)
        result = sddm.sdd_manager_literal(num, self.manager)
        sddm.sdd_ref(result, self.manager)
        if negated:
            newresult = sdd_negate(result, self.manager)
            sddm.sdd_deref(result, self.manager)
            result = newresult
        return result

    def get_literal_value(self, lit):
        """
        Returns an SDD representing the literal value. For each literal this will be (the negation of) an sdd variable
        If an input atom has no sdd var representation yet, a new sdd variable is created
        :param lit: the literal to represent
        :return: An sdd representing this literal.
        """
        if not lit.negated:
            result = self.atom2sdd.setdefault(lit, sddm.sdd_manager_false(self.manager))
            sddm.sdd_ref(result, self.manager)
            return result
        else:
            result = self.atom2sdd.setdefault(lit.negate(), sddm.sdd_manager_false(self.manager))
            result = sdd_negate(result, self.manager)
            # Note: sdd_negate increases the reference count by one.
            return result

    def set_sdd_var_num(self, atom, num):
        """
        Sets the sdd variable associated to atom to be num. Can only be called if atom does not have an image
        yet. Throws an exception otherwise.
        :param atom: The atom to set a number for
        :param num: The number to set the value to
        :return: nothing
        """
        sdd_var_num = self.atom2sddnum.get(atom)
        if sdd_var_num is not None:
            raise Exception("Internal error: Trying to overwrite value of sddvariables")
        self.atom2sddnum[atom] = num

    def get_sdd_var_num(self, atom):
        """
        Returns an sdd variable respresenting atom. Finds this in atom2sddnum if possible, creates a new ID otherwise
        :param atom: The atom to get a number for
        :return: a number such taht after calling this method atom2sddnum[atom] = num
        """
        sdd_var_num = self.atom2sddnum.get(atom)
        if sdd_var_num is None:
            largest_sdd_num = sddm.sdd_manager_var_count(self.manager)
            sdd_var_num = largest_sdd_num + 1
            self.atom2sddnum[atom] = sdd_var_num
            sddm.sdd_manager_add_var_after_last(self.manager)
        return sdd_var_num

    def set_constraint(self, sdd):
        """
        Sets the value of "constraint" to the given sdd.
        Useful for restoring previous state without recompiling
        :param sdd: the given sdd
        :return: nothing
        """
        self.constraint = sdd

    def set_entire_program_as_sdd(self, sdd):
        """
        Sets the value of "_entire_program_as_sdd" to the given sdd.
        Useful for restoring previous state without recompiling
        :param sdd: the given sdd
        :return: nothing
        """
        self._entire_program_as_sdd = sdd


    def get_entire_program_as_sdd(self):
        """
        Returns an SDD representing the entire logic program.
        :return: an SDD representing the entire logic program
        """
        if self._entire_program_as_sdd is not None:
            return self._entire_program_as_sdd

        equivalences = []

        for atom, sdd in self.atom2sdd.items():
            sdd_var_num = self.get_sdd_var_num(atom)
            sdd4atom = sddm.sdd_manager_literal(sdd_var_num, self.manager)
            sddm.sdd_ref(sdd4atom, self.manager)
            equiv = sdd_equiv(sdd4atom, sdd, self.manager)
            sddm.sdd_deref(sdd4atom, self.manager)
            equivalences.append(equiv)

        equivalences.append(self.constraint)
        sddm.sdd_ref(self.constraint, self.manager)

        equivalences = sort_sdds(equivalences)




        result = sdd_conjoin(equivalences, self.manager)



        for sdd in equivalences:
            sddm.sdd_ref(sdd, self.manager)

        self._entire_program_as_sdd = result
        return result

    def get_entire_program_as_sdd_over_voc(self, voc):
        """
        Returns an SDD representing the following expression:
            \exists \vvv: \PP,
        where \PP is the logic program
        \vvv is the set of variables in \PP but not in \voc
        :return: an SDD representing this expression
        """
        prog = self.get_entire_program_as_sdd()

        goodsddvars = []
        for atom in voc:
            goodsddvars.append(self.atom2sddnum[atom])

        to_quantify = []

        nbvars_in_manager = sddm.sdd_manager_var_count(self.manager)
        for i in range(1, nbvars_in_manager + 1):
            if i not in goodsddvars:
                to_quantify.append(i)

        return sdd_exists(to_quantify, prog, self.manager)


class SDDCompiler():
    """Contains the intermediate data strutures for compilation of logic program: mapping from literals to SDDs """

    def __init__(self, args):
        self.args = args

        # Whether or not only the smallest SDDs should be compiled each iteration
        self.compileOnlySmallest = True
        self.param2sddNum = OrderedDict()
        self.lit2sdd = OrderedDict()
        self.manager = None  # Will be initialised in first_iteration

        # Used during unfounded set computations only
        self.backup_sdd = OrderedDict()

    def print_sdd_size(self):
        print("size: ", sddm.sdd_manager_size(self.manager))
        print("count: ", sddm.sdd_manager_count(self.manager))

    def save_sdd(self,name):
        sddm.sdd_save_as_dot()

    def finish_two_valued(self, program):
        """ Finishes the compilation
        :return: a TwoValuedCompiledProgram containing the result of the compilation
        """
        atom2sdd = OrderedDict()
        for lit, sdd in self.lit2sdd.items():
            if lit.negated:
                atom = lit.negate()
                atom_sdd = self.lit2sdd.get(atom)
                if atom_sdd is None:
                    raise Exception("Compiled " + repr(lit) + " but never compiled value for " + repr(atom))
                neg_atom_sdd = sdd_negate(atom_sdd, self.manager)
                if neg_atom_sdd != sdd:
                    raise Exception("Trying to compile two-valued, but the result is three-valued for literal: " + str(
                        lit) + "\n This probably means that there are errors in the input program ("
                               "the well-founded model is not two-valued; check your program for loops over negation")
                sddm.sdd_deref(neg_atom_sdd, self.manager)
            else:
                atom2sdd[lit] = sdd


        return TwoValuedCompiledProgram(atom2sdd, self.param2sddNum, self.manager, program)

    def finish_three_valued(self):
        """ Finishes the compilation in case the result is allowed to be threevalued
        :return: a ThreeValuedCompiledProgram containing the result of the compilation
        """
        # TODO implement!
        self.lit2sdd = self.lit2sdd
        raise NotImplementedError("Three valued finishing")

    def first_iteration(self, parameters):
        """ Initialises the compiler with a set of parameters
        Ensures that for each SDD in the image of lit2sdd, the reference count is one.
        :param parameters: the set of "open" atoms, aka parameters of the logic program
        :return: nothing
        """
        self.manager = sddm.sdd_manager_create(len(parameters), 1)
        largest_atom_num_used = 0
        #print("=========================== In the first iteration: =============================")
        for param in parameters:
            largest_atom_num_used += 1
            self.param2sddNum[param] = largest_atom_num_used
            sdd4lit = sddm.sdd_manager_literal(largest_atom_num_used, self.manager)
            self.lit2sdd[param] = sdd4lit
            sdd4neglit = sdd_negate(sdd4lit, self.manager)
            self.lit2sdd[param.negate()] = sdd4neglit
            sddm.sdd_ref(sdd4lit, self.manager)
            sddm.sdd_ref(sdd4neglit, self.manager)

            #sddm.sdd_save_as_dot(str(largest_atom_num_used),sdd4lit)
            #sddm.sdd_save_as_dot("not_"+str(largest_atom_num_used),sdd4neglit)

        #print(self.lit2sdd,len(self.lit2sdd))
        #print(sddm.sdd_manager_size(self.manager))
        #print("===========================END In the first iteration: =============================")


        return

    def compile_at_least_one_proof(self, new_proofs):
        """ Compiles at least one proof from new_proofs. Updates lit2sdd accordingly
        :param new_proofs: a mapping from heads to formulas such that new_proofs(h) is a new proof for h
        Concretely this means that lit2sdd(h) | compile(new_proofs(h)) will be the new value of lit2sdd(h)
        Removes all compiled proofs from new_proofs
        :return: an array containing all of literals whose sdd has changed in the process (in order)
        """
        changed_heads = []

        sorted_heads = self.sort_proofs(new_proofs)
        for head in sorted_heads:
            sdd_head_changed = self.compile_new_proof(head, new_proofs[head])
            del new_proofs[head]
            if sdd_head_changed:
                changed_heads.append(head)
        return changed_heads

    def compile_at_least_one_disproof(self, new_disproofs):
        """ Compiles at least one disproof from new_disproofs. Updates lit2sdd accordingly
        :param new_disproofs: a mapping from heads to formulas such that new_disproofs(h) is a new disproof for h
        (i.e., a proof for ~h)
        Concretely this means that lit2sdd(h) & compile(new_disproofs(h)) will be the new value of lit2sdd(h)
        Removes all compiled disproofs from new_disproofs
        :return: an array containing all of literals whose sdd has changed in the process (in order)
        """
        changed_heads = []

        sorted_heads = self.sort_proofs(new_disproofs)
        for head in sorted_heads:
            sdd_head_changed = self.compile_new_disproof(head, new_disproofs[head])
            del new_disproofs[head]
            if sdd_head_changed:
                changed_heads.append(head)
        return changed_heads

    def set_value_to(self, lit, formula):
        """
        Sets the value of lit in lit2sdd to the value obtained by compiling the formula
        :param lit: the literal whose value should change
        :param formula: the formula representing the new value of lit
        :return:
        """
        orig_value = self.lit2sdd.get(lit, sddm.sdd_manager_false(self.manager))
        new_value = compile_formula(formula, self.manager, self.get_literal_value)
        sddm.sdd_deref(orig_value, self.manager)
        self.lit2sdd[lit] = new_value

    def sort_proofs(self, proofs):
        """
        Sorts the proofs according to the estimated size of their compilation
        :param proofs: the proofs to sort
        :return: a list of heads whose proofs are small enough to be compiled.
        """
        lit2size = {}  # serves for caching sdd compilation size
        size2head = []  # A mapping from each size to the corresponding head
        for head, newProof in proofs.items():
            this_proof_size = 1
            if head in self.lit2sdd:
                if head not in lit2size:
                    lit2size[head] = smart_size(self.lit2sdd[head])
                this_proof_size *= lit2size[head]
            for lit in newProof.get_indir_subliterals():
                if lit not in lit2size:
                    lit2size[lit] = smart_size(self.lit2sdd[lit])
                this_proof_size *= lit2size[lit]
            size2head.append([this_proof_size, head])

        size2head.sort()
        sorted_heads = []
        compilesize = 0
        for h in size2head:
            if self.compileOnlySmallest and h[0] > compilesize > 0:
                break
            sorted_heads.append(h[1])
            compilesize = h[0]
        return sorted_heads

    def compile_new_proof(self, head, new_proof):
        """
        Compiles a new proof for head
        :param head: the head whose sdd should be updated
        :param new_proof: a formula representing a new proof for head. I.e., a formula such that
        lit2sdd(head) should be updated to lit2sdd(head) | compile(new_proof)
        :return True if lit2sdd[head] has changed; False otherwise
        """

        if head not in self.lit2sdd:
            self.lit2sdd[head] = sddm.sdd_manager_false(self.manager)

        sdd_head_changed = True
        sdd_head_old = self.lit2sdd[head]

        new_sdd = compile_formula(new_proof, self.manager, self.get_literal_value)
        # new_sdd is an SDD represeting the new proof



        final_sdd = sdd_disjoin([self.lit2sdd[head], new_sdd], self.manager)
        # final_sdd is an SDD represeting the new value of lit2sdd[head]
        sddm.sdd_deref(new_sdd, self.manager)
        self.lit2sdd[head] = final_sdd

        if sdd_head_old == self.lit2sdd[head]:
            sdd_head_changed = False
        sddm.sdd_deref(sdd_head_old, self.manager)

        return sdd_head_changed

    def compile_new_disproof(self, head, new_disproof):
        """
        Compiles a new disproof for head
        :param head: the head whose sdd should be updated
        :param new_disproof: a formula representing a new disproof for head. I.e., a formula such that
        lit2sdd(head) should be updated to lit2sdd(head) &  compile(new_disproof)
        :return True if lit2sdd[head] has changed; False otherwise
        """

        sdd_head_changed = True
        sdd_head_old = self.lit2sdd[head]

        new_sdd = compile_formula(new_disproof, self.manager, self.get_literal_value)
        # new_sdd is an SDD represeting the new disproof
        final_sdd = sdd_conjoin([sdd_head_old, new_sdd], self.manager)
        # final_sdd is an SDD represeting the new value of lit2sdd[head]
        sddm.sdd_deref(new_sdd, self.manager)
        self.lit2sdd[head] = final_sdd

        if sdd_head_old == self.lit2sdd[head]:
            sdd_head_changed = False
        sddm.sdd_deref(sdd_head_old, self.manager)

        return sdd_head_changed

    def backup_neg_literals(self):
        """
        Backups all SDDs associated to negative literals and replaces those SDDs by "True".
        This is used to prepare a logic program for unfounded set computation.
        NOTE: if some atom is already interpreted two-valued, its negative literal will not be updated.
        :return: the set of literals whose value has changed.
        """
        changed = []
        self.backup_sdd = OrderedDict()
        for lit, sdd in self.lit2sdd.items():
            to_backup = lit.negated
            to_backup &= (sdd != sddm.sdd_manager_true(self.manager))
            true_sdd = self.lit2sdd.get(lit.negate(), sddm.sdd_manager_false(self.manager))
            neg_true_sdd = sdd_negate(true_sdd, self.manager)
            to_backup &= (sdd != neg_true_sdd)
            sddm.sdd_deref(neg_true_sdd, self.manager)
            if to_backup:
                self.backup_sdd[lit] = sdd
                changed.append(lit)
                self.lit2sdd[lit] = sddm.sdd_manager_true(self.manager)
        return changed

    def collect_changed_lits_since_backup(self):
        """
        Collects all literals whose values have changed since last backup.
        Also cleans the set of backupSDDs
        :return: the set of literals whose value has changed.
        """
        changed = []
        for lit, backup_sdd in self.backup_sdd.items():
            new_sdd = self.lit2sdd[lit]
            if new_sdd != backup_sdd:
                changed.append(lit)
            sddm.sdd_deref(backup_sdd, self.manager)
        self.backup_sdd = OrderedDict()
        return changed

    def compile_default(self, lits):
        """
        Sets the literals in lits to their default value (false) if they are not compiled yet
        :param lits: The set of literals that should have a value after calling this method
        :return:
        """
        for lit in lits:
            self.lit2sdd.setdefault(lit, sddm.sdd_manager_false(self.manager))

    def get_literal_value(self, lit):
        """
        Returns an SDD representing the literal value.
        This basically looks up lit in lit2sdd, returning a false SDD in case this literal
        is not yet present in the mapping
        :param lit: the literal to represent
        :return: An sdd representing this literal. Increases the reference count of this sdd by one
        """
        result = self.lit2sdd.setdefault(lit, sddm.sdd_manager_false(self.manager))
        sddm.sdd_ref(result, self.manager)
        return result

    # =============================
    # INFERENCE METHODS
    # =============================

    def test_garbage_collection(self):
        all_sdds = list(self.lit2sdd.values())
        single_top = sdd_conjoin(all_sdds, self.manager)
        self.print_ref_counts()
        for sdd in all_sdds:
            sddm.sdd_deref(sdd, self.manager)

        sddsize = sddm.sdd_size(single_top)
        sdd_livesize = sddm.sdd_manager_live_size(self.manager)

        if sddsize == sdd_livesize:
            print('Your dereferencing was completely correct!')
        else:
            print('WARNING : Your dereferencing was NOT correct !!')
        return

    def free_manager(self):
        sddm.sdd_manager_free(self.manager)
        return

    def get_sdd_size(self):
        return sddm.sdd_manager_live_size(self.manager)

# HELP and TEST functions
# =======================
    def print_ref_counts(self):
        for head, sdd in self.lit2sdd.items():
            print(repr(head) + " --> " + str(sddm.sdd_ref_count(sdd)))


def sorted_apply(sdd_list, manager, apply_func):
    """
    Performs an apply operation on a list of sdds. Does this sorted in the sense that two smaller sdds will be
    applied before applying to larger sdds
    :param sdd_list: the sdds
    :param manager: the manager
    :param apply_func: an apply function (conjoin or disjoin); this function must increase reference count of its result
    :return: An SDD: the result of the apply. This SDD has +1 reference count.
    """
    for sdd in sdd_list:
        sddm.sdd_ref(sdd, manager)
    while len(sdd_list) > 2:
        sdd_list = sort_sdds(sdd_list)
        sdd1 = sdd_list[0]
        sdd2 = sdd_list[1]
        new = apply_func([sdd1, sdd2], manager)
        sdd_list.remove(sdd1)
        sddm.sdd_deref(sdd1, manager)
        sdd_list.remove(sdd2)
        sddm.sdd_deref(sdd2, manager)
        sdd_list.append(new)

    result = apply_func(sdd_list, manager)
    for sdd in sdd_list:
        sddm.sdd_deref(sdd, manager)
    return result






class Timeout():
    """Timeout class using ALARM signal."""

    class Timeout(Exception):
        pass

    def __init__(self, sec):
        self.sec = sec

    def __enter__(self):
        signal.signal(signal.SIGALRM, self.raise_timeout)
        signal.alarm(self.sec)

    def __exit__(self, *args):
        signal.alarm(0)  # disable alarm

    def raise_timeout(self, *args):
        raise Timeout.Timeout()



def sdd_conjoin(sdd_list, manager):
    """
    Conjoins all the input sdds to one large sdd
    :param sdd_list: a list of input sdds
    :param manager: the manager managing all of the input sdds
    :return: An SDD: the conjunction of the input sdds. This SDD has +1 reference count.
    """

    #print("lenth of sdd_list = ",len(sdd_list))


    ####try divide and conquer, auto_SDD minimization
    '''if sdd_list is None:
        print("sdd_list is None")
        exit(0)
    elif len(sdd_list) == 1:
        alpha = sddm.sdd_manager_true(manager)
        beta = sddm.sdd_conjoin(alpha,sdd_list[0],manager)
        sddm.sdd_ref(beta, manager)
        sddm.sdd_deref(alpha, manager)
        alpha = beta
        sddm.sdd_ref(sdd_list[0], manager)

        return sdd_list[0]
    else:
        mid = len(sdd_list)//2
        left = sdd_conjoin(sdd_list[:mid], manager)
        right = sdd_conjoin(sdd_list[mid:], manager)


        bigger =  sddm.sdd_conjoin(left, right, manager)
        sddm.sdd_deref(left, manager)
        sddm.sdd_deref(right, manager)
        sddm.sdd_ref(bigger, manager)

        return bigger'''


    alpha = sddm.sdd_manager_true(manager)
    for sdd in sdd_list:
        beta = sddm.sdd_conjoin(alpha, sdd, manager)
        sddm.sdd_ref(beta, manager)
        sddm.sdd_deref(alpha, manager)
        alpha = beta

    return alpha

def sdd_disjoin(sdd_list, manager):
    """
    Disjoins all the input sdds to one large sdd
    :param sdd_list: a list of input sdds
    :param manager: the manager managing all of the input sdds
    :return: An SDD: the disjunction of the input sdds. This SDD has +1 reference count.
    """

    import time
    alpha = sddm.sdd_manager_false(manager)
    for sdd in sdd_list:
        beta = sddm.sdd_disjoin(alpha, sdd, manager)
        sddm.sdd_ref(beta, manager)
        sddm.sdd_deref(alpha, manager)
        alpha = beta
    return alpha


def sdd_negate(sdd, manager):
    """
    Negates the input sdd
    :param sdd: The sdd to negate
    :param manager: the manager managing the input sdd
    :return: An SDD: the negation of the input sdd. This SDD has +1 reference count.
    """
    result = sddm.sdd_negate(sdd, manager)
    sddm.sdd_ref(result, manager)
    return result


def sdd_equiv(sdd1, sdd2, manager):
    """
    Returns the sdd: sdd1 <=> sdd2
    :param sdd1: An input SDD
    :param sdd2: An input SDD
    :param manager: the manager managing the input sdds
    :return: An SDD: the sdd representing sdd1 <=> sdd2. This SDD has +1 reference count.
    """

    """
    sdd1n = sdd_negate(sdd1, manager)
    sddm.sdd_ref(sdd1n, manager)
    sdd2n = sdd_negate(sdd2, manager)
    sddm.sdd_ref(sdd2n, manager)
    sdde1 = sdd_conjoin([sdd1, sdd2], manager)
    sddm.sdd_ref(sdde1, manager)
    sdde2 = sdd_conjoin([sdd1n, sdd2n], manager)
    sddm.sdd_ref(sdde2, manager)
    sddequiv = sdd_disjoin([sdde1, sdde2], manager)
    sddm.sdd_ref(sddequiv, manager)
    """

    sdd1n = sdd_negate(sdd1, manager)
    sddm.sdd_ref(sdd1n, manager)
    sdd2n = sdd_negate(sdd2, manager)
    sddm.sdd_ref(sdd2n, manager)
    sdde1 = sdd_disjoin([sdd1, sdd2n], manager)
    sddm.sdd_ref(sdde1, manager)
    sdde2 = sdd_disjoin([sdd1n, sdd2], manager)
    sddm.sdd_ref(sdde2, manager)
    sddequiv = sdd_conjoin([sdde1, sdde2], manager)
    sddm.sdd_ref(sddequiv, manager)

    sddm.sdd_deref(sdd1n, manager)
    sddm.sdd_deref(sdd2n, manager)
    sddm.sdd_deref(sdde1, manager)
    sddm.sdd_deref(sdde2, manager)
    return sddequiv


def sdd_exists(variables, sdd, manager):
    """
    Returns the formula
        \exist v_1, v_2, ..., v_n: sdd
    where v_i are the vars in vars
    :param variables: variables to quantify existentially over
    :param sdd: the sdd to quantify
    :param manager: a manager managing both the sdd and vars
    :return: The resulting SDD. This SDD has +1 reference count
    """
    result = sdd
    sddm.sdd_ref(result, manager)
    for var in variables:
        newresult = sddm.sdd_exists(var, result, manager)
        sddm.sdd_ref(newresult, manager)
        sddm.sdd_deref(result, manager)
        result = newresult
    return result


def sdd_forall(variables, sdd, manager):
    """
    Returns the formula
        \forall v_1, v_2, ..., v_n: sdd
    where v_i are the vars in vars
    :param variables: variables to quantify universally over
    :param sdd: the sdd to quantify
    :param manager: a manager managing both the sdd and vars
    :return: The resulting SDD. This SDD has +1 reference count
    """
    result = sdd
    sddm.sdd_ref(result, manager)
    for var in variables:
        newresult = sddm.sdd_forall(var, result, manager)
        sddm.sdd_ref(newresult, manager)
        sddm.sdd_deref(result, manager)
        result = newresult
    return result


def compile_formula(formula, manager, get_literal_value):

    """
    Compiles the input formula for the given manager.
    :param formula: the logical formula to compile to an sdd
    :param manager: the manager managing all sdds
    :param get_literal_value: a function that returns for each literal an sdd, used to compile leafs of the formula
            this function is supposed to increase the reference count of the SDD it outputs by one!
    :return: An SDD represeting the formula. This SDD has +1 reference count.
    """
    # Flyweights doen iets raar met types. type(formula) is ground.LIterals werkt bijvoorbeeld niet.
    # Daarom de lelijke hacks
    if formula is logic.TrueForm():
        return sddm.sdd_manager_true(manager)
    elif formula is logic.FalseForm():
        return sddm.sdd_manager_false(manager)
    elif isinstance(formula, type(logic.Literal("", True))):
        result = get_literal_value(formula)
        # Note: get_literal_value does the increment of the reference count
        return result
    elif type(formula) is logic.Disjunction:
        compiled_subforms = []
        for subform in formula.subforms:
            compiled_subform = compile_formula(subform, manager, get_literal_value)
            compiled_subforms.append(compiled_subform)
        result = sdd_disjoin(compiled_subforms, manager)
        # Note: sdd_disjoin does the increment of the reference count
        for compiled_form in compiled_subforms:
            sddm.sdd_deref(compiled_form, manager)
        return result
    elif type(formula) is logic.Conjunction:
        compiled_subforms = []
        for subform in formula.subforms:
            compiled_subform = compile_formula(subform, manager, get_literal_value)
            compiled_subforms.append(compiled_subform)
        result = sdd_conjoin(compiled_subforms, manager)
        # Note: sdd_conjoin does the increment of the reference count
        for compiled_form in compiled_subforms:
            sddm.sdd_deref(compiled_form, manager)
        return result
    else:
        print(type(formula))
        raise NotImplementedError


def compute_model_count(sdd, manager):
    return sddm.sdd_model_count(sdd, manager)


def compute_weighted_model_count(sdd, manager, p_weights=None,e_weights=None,q_atom = None):

    #Create new WMC manager and set the log_mode = true
    wmc_manager = sddm.wmc_manager_new(sdd, 1, manager)

    if p_weights is not None:
        for key, weight in p_weights.items():
            sddm.wmc_set_literal_weight(key, weight, wmc_manager)

    if e_weights is not None:
        for key, weight in e_weights.items():
            sddm.wmc_set_literal_weight(key, -float("inf"), wmc_manager)


    # Need query to be always true, set the negation of query atom to be negative infinity
    if q_atom is not None:
        sddm.wmc_set_literal_weight(q_atom,-float("inf") , wmc_manager)

    wmc = sddm.wmc_propagate(wmc_manager)
    sddm.wmc_manager_free(wmc_manager)

    return wmc


def sort_sdds(sdds):
    """
    Sorts the sdds according to their size
    :param sdds: the sdds to sort
    :return: a sorted list of sdds.
    """
    size_sdd = []  # A list of tuples (size,sdd)
    for sdd in sdds:
        size = smart_size(sdd)
        size_sdd.append([size, sdd])

    size_sdd.sort( key=lambda t: t[0])
    sorted_sdds = []
    for h in size_sdd:
        sorted_sdds.append(h[1])
    return sorted_sdds


def smart_size(sdd):
    """
    SDD size that differentiates between trivial sdds and literals
    :param sdd: the sdd to compute the size of
    :return: an adapted size of the given sdd
    """
    if sddm.sdd_node_is_false(sdd):
        return 1
    elif sddm.sdd_node_is_true(sdd):
        return 1
    elif sddm.sdd_node_is_literal(sdd):
        return 2
    else:
        return sddm.sdd_size(sdd)
