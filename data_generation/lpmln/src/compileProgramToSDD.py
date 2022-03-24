from src import compile
from src import logic
from src import transform
from lib import sdd as sddm
from collections import *
import signal

# ========================
# INTERFACE:
# ========================


def convert_program_to_sdd(args, program):
    if args.verbosity > 0:
        print(">>>Start transforming program (symbolic Gelfond-Lifschitz reduct)")

    program, varmap = transform.transform(program)

    if args.verbosity > 0:
        print(">>>Done transforming program")
    if args.verbosity > 4:
        print("Transformed logic program:")
        print(program)
    compiler = ProgramToSDDCompiler(args, program)
    result = compiler.run()
    if varmap is not None:
        for var, copy in varmap.items():
            sddnum = result.get_sdd_var_num(copy)
            result.set_sdd_var_num(var, sddnum)

    return result

'''
#  ========================
#   INTERNALS
#  ========================

class CompletionCompiler:
    def __init__(self, args, program):
        self.args = args
        self.program = program

    def run(self):
        manager = sddm.sdd_manager_create(1, 1)
        internal_lit = logic.Literal("internal_true", False)
        # This internal literal serves for avoiding the bug that no manager without variables can be created
        result = compile.TwoValuedCompiledProgram({}, {internal_lit: 1}, manager,
                                                  self.program)
        result.addconstraint_without_evaluating(internal_lit)
        # To restore the model count broken by adding one extra variable

        constraint = self.program.get_completion()
        result.addconstraint_without_evaluating(constraint)

        result.addconstraint_without_evaluating(self.program.constraint)
        return result
'''

class ProgramToSDDCompiler:
    def __init__(self, args, program):
        self.args = args
        self.orig_program = program
        self.rules = list(program.rules)
        self.constraint = program.constraint
        self.parameters = program.parameters
        self.Compiler = compile.SDDCompiler(self.args)
        self.dependency = OrderedDict()

        # For verbosity purposes onlys
        self.iteration = 0

        # The following two variables serve specifically for usage during unfounded set computations (Well-founded
        # semantics)
        self.negated_orig_rules = []
        self.negated_dependency = OrderedDict()

    def run(self):

        if self.args.timeout is not None:
            signal.alarm(int(self.args.timeout))
            signal.signal(signal.SIGALRM, self.on_timeout)
        try:
            self.compile_rules()
            result = self.Compiler.finish_two_valued(self.orig_program)
            result.addconstraint(self.constraint)
            return result
        except (TimeoutException, SystemError) as e:
            if e.args[0] == "Process timeout":
                # HACK: SystemError is thrown by sdd library on timeout
                # We identify these exception by means of their message... 
                print('TIMEOUT before compiling rules was finished!! Performing inference on approximation.')
                result = self.Compiler.finish_three_valued()
                result.addconstraint(self.constraint)
                return result
            else:
                raise e

    @staticmethod
    def on_timeout(*args):
        raise TimeoutException("Process timeout")

    def compile_rules(self):
        self.Compiler.first_iteration(self.parameters)

        # contains negation is true if and only if at least one defined atom is negated.
        contains_negation = self.check_negation()


        if contains_negation:
            self.add_negation_rules()

        if self.args.verbosity > 8:
            print("Internal rules:")
            print(self.rules)

        # dependency contains a mapping from every literal to the set of rules it occurs in the body of.
        self.analyze_dependency()

        changed_lits = []


        for param in self.parameters:
            changed_lits.append(param)
            changed_lits.append(param.negate())

        changed_lits.extend(self.compile_trivial_rules())

        # compiled_lits simply contains a list of all lits that have been compiled up until now
        compiled_lits = set()
        compiled_lits.update(changed_lits)
        first_time = True



        while len(changed_lits) > 0:
            self.forward_iteration(changed_lits, compiled_lits, first_time)
            if contains_negation:
                if first_time:
                    # This code contains an optimization: during the first forward iteration, not all literals have been
                    # compiled. In this iteration, an "uncompiled" literal is assumed to be false. However, starting
                    # from the unfounnded set computation, such an assumption can no longer be made.
                    self.ensure_all_compiled()
                    first_time = False
                changed_lits = self.Compiler.backup_neg_literals()
                self.ufs_iteration(changed_lits)
                changed_lits = self.Compiler.collect_changed_lits_since_backup()

        if self.args.verbosity > 4:
            print("Finished compiling rules. Wrapping up compiler")

    def compile_trivial_rules(self):
        """
        Compiles all trivial rules (i.e., with a true/false body) and removes them from the rule set
        :return: the set of all literals changed by this action
        """
        changed = []
        toremove = []
        for rule in self.rules:
            if rule.body.is_true():
                self.Compiler.set_value_to(rule.head, logic.TrueForm())
                self.Compiler.set_value_to(rule.head.negate(), logic.FalseForm())
                toremove.append(rule)
                changed.append(rule.head)
            elif rule.body.is_false():
                toremove.append(rule)
        return changed

    def forward_iteration(self, changed_lits, compiled_lits, first_time):
        # new_proofs maps every atom to a formula that represents the new justifications for this atom.
        # In some next step of the iteration, the value of lit should be updated to its current value OR new_proofs[lit]
        new_proofs = OrderedDict()

        while len(changed_lits) > 0 or len(new_proofs) > 0:

            self.iteration += 1
            if self.args.verbosity > 3:
                print("Compiler iteration: " + str(self.iteration))
                print("- #Proofs in queue: " + str(len(new_proofs)))
                print("- #Changed literals in queue: " + str(len(changed_lits)))
            if self.args.verbosity > 7:
                print("- Changed literals in queue:")
                print(changed_lits)
            self.forward_step(compiled_lits, changed_lits, new_proofs, first_time)
            if self.args.verbosity > 8:
                print("- Proofs in queue:")
                print(new_proofs)
            more_changed_lits = self.Compiler.compile_at_least_one_proof(new_proofs)
            changed_lits.extend(more_changed_lits)
            if self.args.verbosity > 5:
                print("The following literals where changed in this iteration:")
                print(changed_lits)
            compiled_lits.update(changed_lits)



    def ufs_iteration(self, changed_lits):

        # new_disproofs maps every atom to a formula that represents new justifications for this atom.
        # In some next step of the iteration, the value of lit should be updated to newProofs[lit]
        new_disproofs = OrderedDict()

        # First of all: all rules of atoms that were arbitrarily changed might need to be reevaluated
        # (in order to re-discover some previously known disproofs)
        for rule in self.negated_orig_rules:
            if rule.head in changed_lits:
                prev = new_disproofs.get(rule.head, logic.TrueForm())
                new_disproofs[rule.head] = prev.conjoin(rule.body)
        while len(changed_lits) > 0 or len(new_disproofs) > 0:
            self.iteration += 1
            if self.args.verbosity > 3:
                print("Compiler iteration (UFS): " + str(self.iteration))
                print("- Disproofs in queue: " + str(len(new_disproofs)))
                print("- Changed literals in queue: " + str(len(changed_lits)))
            if self.args.verbosity > 7:
                print("- Changed literals in queue:")
                print(changed_lits)
            self.ufs_step(changed_lits, new_disproofs)
            if self.args.verbosity > 8:
                print("Disproofs that remain to be compiled:")
                print(new_disproofs)
            more_changed_lits = self.Compiler.compile_at_least_one_disproof(new_disproofs)
            changed_lits.extend(more_changed_lits)
            if self.args.verbosity > 5:
                print("The following literals where changed in this iteration:")
                print(changed_lits)

    def forward_step(self, compiled_lits, changed_lits, new_proofs, some_uncompiled):

        for lit in changed_lits:
            for rule in self.dependency.setdefault(lit, []):
                if not some_uncompiled or set(rule.body.subforms).issubset(compiled_lits):
                    prev_proof = new_proofs.setdefault(rule.head, logic.FalseForm())
                    new_proofs[rule.head] = prev_proof.disjoin(rule.body)


        changed_lits.clear()


    def ufs_step(self, changed_lits, new_disproofs):
        for lit in changed_lits:
            for rule in self.negated_dependency.setdefault(lit, []):
                prev_proof = new_disproofs.setdefault(rule.head, logic.TrueForm())
                new_disproofs[rule.head] = prev_proof.conjoin(rule.body)
        changed_lits.clear()

    def analyze_dependency(self):
        self.dependency = OrderedDict()
        for rule in self.rules:
            for lit in rule.body.subforms:
                self.dependency.setdefault(lit, []).append(rule)

        self.negated_dependency = OrderedDict()
        for rule in self.negated_orig_rules:
            for lit in rule.body.subforms:
                self.negated_dependency.setdefault(lit, []).append(rule)


    def check_negation(self):
        defined_atoms = set()
        neg_atoms = set()  # atoms occuring under negation
        for rule in self.rules:
            defined_atoms.add(rule.head)
            for lit in rule.body.get_indir_subliterals():
                if lit.negated:
                    neg_atoms.add(lit.negate())
        return not defined_atoms.isdisjoint(neg_atoms)

    def add_negation_rules(self):
        """
        Extends self.rules with rules for negated literals
        Also sets self.negated_orig_rules to its correct value
        :return: nothing
        """
        pos_unique_head_rules = OrderedDict()
        self.negated_orig_rules = []
        for rule in self.rules:
            head = rule.head
            body = rule.body
            head_n = head.negate()
            body_n = body.negate()
            prev = pos_unique_head_rules.get(head, logic.FalseForm())
            pos_unique_head_rules[head] = prev.disjoin(body)
            self.negated_orig_rules.append(logic.Rule(head_n, body_n))
        for head, body in pos_unique_head_rules.items():
            negrule = logic.Rule(head.negate(), body.negate())
            self.rules.append(negrule)

    def ensure_all_compiled(self):
        lits = []
        for rule in self.rules:
            if rule.head not in lits:
                lits.append(rule.head)
                lits.append(rule.head.negate())
            for lit in rule.body.get_indir_subliterals():
                if lit not in lits:
                    lits.append(lit)
                    lits.append(lit.negate())
        self.Compiler.compile_default(lits)


class TimeoutException(Exception):
    pass
