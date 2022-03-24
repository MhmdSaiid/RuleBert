from collections import *
import sys
from src.logic import *

# ========================
# INTERFACE:
# ========================


def parse(in_file):
    parser = LparseParser(in_file)
    return parser.parse()



#  ========================
#   INTERNALS:
#  ========================

class Parser():
    def __init__(self, in_file):
        self.inFile = in_file
        self.rules = []
        self.constraint = TrueForm()
        self.symbolTable = OrderedDict()
        self.parameters = []
        self.weights = OrderedDict()

    def finish(self):
        """
        Finishes parsing and returns the parsed object
        :return: a logic program
        :rtype : LogicProgram
        """
        return LogicProgram(self.rules, self.constraint, self.parameters, self.symbolTable, self.weights)

    def parse(self):
        if self.inFile is None:
            return self.internal_parse(sys.stdin)
        else:
            return self.internal_parse(self.inFile)

    def internal_parse(self, f_in):
        """
        The pure virtual internal parse method. To be implemented by each subclass
        :rtype : LogicProgram
        """
        raise NotImplementedError("Calling pure virtual method in parser")


class LparseParser(Parser):
    def __init__(self, in_file):
        Parser.__init__(self, in_file)

    def internal_parse(self, f_in):

        self.parse_rules(f_in)
        self.parse_symbol_table(f_in)
        self.parse_constraints(f_in)
        self.parse_until_parameters(f_in)
        self.parse_parameters(f_in)
        return self.finish()

    @staticmethod
    def parse_until_parameters(f_in):
        for line in f_in:
            # The 'P' is only for backwards compatibility wrt older versions of this tool. could in principle be removed
            if len(line) > 0 and (line[0] == 'E' or line[0] == 'P'):
                break

    def parse_parameters(self, f_in):
        for line in f_in:
            if line[0] == '0':
                break
            elif line[0] == '%':
                continue
            else:
                line = line.rstrip('\n')
                atom_nb = int(line)
                lit = Literal(atom_nb, False)
                self.parameters.append(lit)

    def parse_constraints(self, f_in):
        for line in f_in:
            if line[0] == '0':
                continue
            elif line[0] == '%':
                continue
            elif line[0:2] == 'B+':
                self.parse_constraints_only(f_in, False)
                break
            else:
                raise UnexpectedLineException("Unexpected line (expecting B+): " + line)

        for line in f_in:
            if line[0] == '0':
                continue
            elif line[0] == '%':
                continue
            elif line[0:2] == 'B-':
                self.parse_constraints_only(f_in, True)
                break
            else:
                raise UnexpectedLineException("Unexpected line (expecting B-): " + line)

        return

    def parse_constraints_only(self, f_in, negated):
        for line in f_in:
            if line[0] == '0':
                break
            elif line[0] == '%':
                continue
            else:
                line = line.rstrip('\n')
                atom_nb = int(line)
                lit = Literal(atom_nb, negated)
                self.constraint = self.constraint.conjoin(lit)

    def parse_symbol_table(self, f_in):
        for line in f_in:
            line = line.rstrip('\n')
            if line[0] == '0':
                break
            if line[0] == '%':
                continue
            else:
                num, name = line.split(None, 1)
                num = int(num)
                name = name.strip()
                self.symbolTable[num] = name

    def parse_rules(self, f_in):
        for line in f_in:
            line = line.rstrip('\n')
            if line[0] == '0':
                break
            if line[0] == '%':
                continue
            rule_type, rest = line.split(None, 1)
            rule_type = int(rule_type)
            if rule_type == 0:
                break
            elif rule_type == 1:
                self.parse_rule(rest)
            elif rule_type == 6:
                print("Warning: Ignoring optimisation statement")
                continue

    def parse_rule(self, line):
        head, body_text = line.split(None, 1)
        head = int(head)
        nb_lits, rest = body_text.split(None, 1)
        nb_lits = int(nb_lits)
        if nb_lits == 0:
            body = TrueForm()
            rule = Rule(Literal(head, False), body)
            self.rules.append(rule)
            return
        nb_negs, rest = rest.split(None, 1)
        nb_negs = int(nb_negs)
        atom_nbs = rest.split(None)
        lits = []
        for i in range(0, len(atom_nbs)):
            atom = int(atom_nbs[i])
            lit = Literal(atom, i < nb_negs)
            lits.append(lit)
        body = Conjunction(lits)
        rule = Rule(Literal(head, False), body)
        self.rules.append(rule)


class UnsupportedRuleTypeException(Exception):
    pass


class UnexpectedLineException(Exception):
    pass
