# =======================
# PROPOSITIONAL FORMULAS
#  =======================


class Formula:
    def negate(self):
        """ Returns the negation of this formula """
        raise NotImplementedError("Taking negation of an abstract formula")

    def conjoin(self, other_formula):
        """ Returns the conjunction of self and other_formula """
        return Conjunction([self, other_formula])

    def disjoin(self, other_formula):
        """ Returns the disjunction of self and other_formula """
        return Disjunction([self, other_formula])

    def get_indir_subliterals(self):
        """ Returns a set() of literals, namely all literals occuring in this formula (or in subformulas) """
        raise NotImplementedError("Indirect subliterals of an abstract formula")

    def equivalent(self, other_formula):
        """
        Makes the equivalence between this formula and the other formula and returns it
        :param other_formula: the formula that should be equivalent to this one
        :return: the formula self <=> other_formula
        """
        conj1 = self.conjoin(other_formula)
        conj2 = self.negate().conjoin(other_formula.negate())
        return conj1.disjoin(conj2)

    def is_true(self):
        """
        Returns true if and only if this is a trivial formula
        :return: true if and only if this is a trivial formula
        """
        return False

    def is_false(self):
        """
        Returns true if and only if this is a trivially false formula
        :return: true if and only if this is a trivially false formula
        """
        return False


class FalseForm(Formula):
    """ A formula representing the logical "false"
    Singleton class"""

    _instance = None

    def __new__(cls):
        if FalseForm._instance is None:
            FalseForm._instance = object.__new__(cls)
        return FalseForm._instance

    def __init__(self):
        self.subforms = []
        return

    def disjoin(self, other_formula):
        return other_formula

    def conjoin(self, other_formula):
        return self

    def negate(self):
        return TrueForm()

    def get_indir_subliterals(self):
        return set()

    def __repr__(self):
        return "false"

    def is_false(self):
        return True


class TrueForm(Formula):
    """ A formula representing the logical "true"
    Singleton class"""

    _instance = None

    def __new__(cls):
        if TrueForm._instance is None:
            TrueForm._instance = object.__new__(cls)
        return TrueForm._instance

    def __init__(self):
        self.subforms = []
        return

    def negate(self):
        return FalseForm()

    def conjoin(self, other_formula):
        return other_formula

    def disjoin(self, other_formula):
        return self

    def get_indir_subliterals(self):
        return set()

    def __repr__(self):
        return "true"

    def is_true(self):
        return True


class Literal(Formula):
    """ A literal: an atom or its negation. The underlying atom is represted by a number or a string...
     (generally: external atoms are numbers, internal atoms are of the form i_n, where n is a number)
     The implementation of the __new__ ensures this is flyweight class"""
    _instances = {}

    def __new__(cls, atomnumber, negated):
        if atomnumber not in Literal._instances:
            Literal._instances[atomnumber] = {}

        if negated not in Literal._instances[atomnumber]:
            obj = object.__new__(cls)
            Literal._instances[atomnumber][negated] = obj
        return Literal._instances[atomnumber][negated]

    def __init__(self, atomnumber, negated):
        self.subforms = []
        self.atomnumber = atomnumber
        self.negated = negated

    def __repr__(self):
        if self.negated:
            return "~ " + str(self.atomnumber)
        else:
            return str(self.atomnumber)

    def negate(self):
        return Literal(self.atomnumber, not self.negated)

    def __lt__(self, other):
        if str(self.atomnumber) < str(other.atomnumber):
            return True
        if self.atomnumber == other.atomnumber:
            return self.negated < other.negated
        return False

    def __gt__(self, other):
        return other.__lt__(self)

    def __le__(self, other):
        return self.__lt__(other) or self == other

    def __ge__(self, other):
        return self.__gt__(other) or self == other

    def get_indir_subliterals(self):
        result = set()
        result.add(self)
        return result

    def __getnewargs__(self):
        return self.atomnumber, self.negated


class Disjunction(Formula):
    def __init__(self, subforms):
        if subforms is None:
            raise Exception("Empty disjunction")
        self.subforms = subforms

    def negate(self):
        newsubf = []
        for form in self.subforms:
            newform = form.negate()
            newsubf.append(newform)
        return Conjunction(newsubf)

    def disjoin(self, other_formula):
        new_forms = list(self.subforms)
        new_forms.append(other_formula)
        return Disjunction(new_forms)

    def __repr__(self):
        if len(self.subforms) == 0:
            return "true"
        answer = ""
        if len(self.subforms) > 1:
            answer += "("
        for form in self.subforms:
            answer += repr(form) + " | "
        answer = answer[0:-3]
        if len(self.subforms) > 1:
            answer += ")"
        return answer

    def get_indir_subliterals(self):
        result = set()

        for form in self.subforms:
            result = result.union(form.get_indir_subliterals())
        return result

    def is_true(self):
        """
        Returns true if and only if this is a trivial formula
        :return: true if and only if this is a trivial formula
        """
        return len(self.subforms) == 0


class Conjunction(Formula):
    def __init__(self, subforms):
        self.subforms = subforms

    def negate(self):
        newsubf = []
        for form in self.subforms:
            newform = form.negate()
            newsubf.append(newform)
        return Disjunction(newsubf)

    def conjoin(self, other_formula):
        new_forms = list(self.subforms)
        new_forms.append(other_formula)
        return Conjunction(new_forms)

    def __repr__(self):
        if len(self.subforms) == 0:
            return "false"
        answer = ""
        if len(self.subforms) > 1:
            answer += "("
        for form in self.subforms:
            answer += repr(form) + " & "
        answer = answer[0:-3]
        if len(self.subforms) > 1:
            answer += ")"
        return answer

    def get_indir_subliterals(self):
        result = set()
        for form in self.subforms:
            result = result.union(form.get_indir_subliterals())
        return result

    def is_false(self):
        return len(self.subforms) == 0


#  =======================
#    LOGIC PROGRAMS
#  =======================

class Rule:
    def __init__(self, head, body):
        self.head = head
        self.body = body
        return

    def __repr__(self):
        return repr(self.head) + " <- " + repr(self.body)


class LogicProgram:
    def __init__(self, rules, constraint, parameters, symbol_table, weights):
        self.rules = rules
        self.parameters = parameters
        self.constraint = constraint
        # SymbolTable maps numbers to names! Not internal data structures
        self.symbolTable = symbol_table
        self.weights = weights
        return

    def __repr__(self):
        result = "***Rules : \n"
        for rule in self.rules:
            result += repr(rule) + "\n"
        result += "***Constraint:\n"
        result += str(self.constraint)
        result += "\n***Symbol Table:\n" + repr(self.symbolTable)
        result += "\n***Parameters:"
        result += repr(self.parameters)
        return result

    def get_completion(self):
        defrules = {}
        for rule in self.rules:
            prev = defrules.get(rule.head, FalseForm())
            new = prev.disjoin(rule.body)
            defrules[rule.head] = new
        result = TrueForm()
        for head, body in defrules.items():
            more = head.equivalent(body)
            result = result.conjoin(more)
        return result

    def get_outvoc(self):
        """
        Returns the output vocabulary associated to this logic program. That is: the list of all atoms that have an
        entry in the symbol table
        :return: a list containing all atoms that have an entry in the symbol table
        """
        result = []
        for num in self.symbolTable.keys():
            result.append(Literal(num, False))
        return result


