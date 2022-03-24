"""
This module contains the basic transformer to traverse and change an AST as
well as constants used during translation.

Classes:
Transformer        -- Base class to modify ASTs.
ProgramTransformer -- Class to rewrite parity constraints.

Functions:
str_location -- Turn a location into a string.

Constants:
g_aux_name -- The name for auxiliary predicates used in the translation.
"""

from clingo import ast as _ast
import clingo as _clingo

# {{{1 basic functions and constants

g_aux_name = "__parity"

def str_location(loc):
    """
    This function takes a location from a clingo AST and transforms it into a
    readable format.
    """
    begin = loc["begin"]
    end   = loc["end"]
    ret = "{}:{}:{}".format(begin["filename"], begin["line"], begin["column"])
    dash = True
    eq = begin["filename"] == end["filename"]
    if not eq:
        ret += "{}{}".format("-" if dash else ":", end["filename"])
        dash = False
    eq = eq and begin["line"] == end["line"]
    if not eq:
        ret += "{}{}".format("-" if dash else ":", end["line"])
        dash = False
    eq = eq and begin["column"] == end["column"]
    if not eq:
        ret += "{}{}".format("-" if dash else ":", end["column"])
        dash = False
    return ret


# {{{1 parse_raw_formula

class Transformer:
    """
    Basic visitor to traverse and modify an AST.

    Transformers to modify an AST should subclass this class and add visit_TYPE
    methods where TYPE corresponds to an ASTType. This function is called
    whenever a node of the respective type is visited. Its return value will
    replace the node in the parent.

    Function visit should be called on the root of the AST to be visited. It is
    the users responsibility to visit children of nodes that have node-specific
    visitor.
    """
    def visit_children(self, x, *args, **kwargs):
        """
        Visits and transforms the children of the given node.
        """
        for key in x.child_keys:
            setattr(x, key, self.visit(getattr(x, key), *args, **kwargs))
        return x

    def visit(self, x, *args, **kwargs):
        """
        Visits the given node and returns its transformation.

        If there is a matching visit_TYPE function where TYPE corresponds to
        the ASTType of the given node then this function called and its value
        returned. Otherwise, its children are visited and transformed.

        This function accepts additional positional and keyword arguments,
        which are passed to node-specific visit functions and to the visit
        function called for child nodes.
        """
        if hasattr(x, "type"):
            attr = "visit_" + str(x.type)
            if hasattr(self, attr):
                return getattr(self, attr)(x, *args, **kwargs)
            else:
                return self.visit_children(x, *args, **kwargs)
        elif isinstance(x, list):
            return [self.visit(y, *args, **kwargs) for y in x]
        elif x is None:
            return x
        else:
            raise TypeError("unexpected type")

    def __call__(self, x, *args, **kwargs):
        """
        Alternative way to call visit.
        """
        return self.visit(x, *args, **kwargs)

class TheoryParser:
    """
    Parser for theory terms mimicking clingo's theory grammars.

    Constants:
    unary  -- Boolean to mark unary operators.
    binary -- Boolean to mark unary operators.
    left   -- Boolean to mark left associativity.
    right  -- Boolean to mark right associativity.
    """
    unary, binary = True, False
    left,  right  = True, False
    table = {
        ("-"  , unary):  (1, None),
        ("+"  , binary): (0, left),
        ("-"  , binary): (0, left), }

    def __init__(self):
        """
        Initializes the parser.
        """
        self.__stack  = []

    def __priority_and_associativity(self, operator):
        """
        Get priority and associativity of the given binary operator.
        """
        return self.table[(operator, self.binary)]

    def __priority(self, operator, unary):
        """
        Get priority of the given unary or binary operator.
        """
        return self.table[(operator, unary)][0]

    def __check(self, operator):
        """
        Returns true if the stack has to be reduced because of the precedence
        of the given binary operator is lower than the preceeding operator on
        the stack.
        """
        if len(self.__stack) < 2:
            return False
        priority, associativity = self.__priority_and_associativity(operator)
        previous_priority       = self.__priority(*self.__stack[-2])
        return previous_priority > priority or (previous_priority == priority and associativity)

    def __reduce(self):
        """
        Combines the last unary or binary term on the stack.
        """
        b = self.__stack.pop()
        operator, unary = self.__stack.pop()
        if unary:
            self.__stack.append(_ast.TheoryFunction(b.location, operator, [b]))
        else:
            a = self.__stack.pop()
            l = {"begin": a.location["begin"], "end": b.location["end"]}
            self.__stack.append(_ast.TheoryFunction(l, operator, [a, b]))

    def parse(self, x):
        """
        Parses the given unparsed term, replacing it by nested theory
        functions.
        """
        del self.__stack[:]
        unary = True
        for element in x.elements:
            for operator in element.operators:
                if not (operator, unary) in self.table:
                    raise RuntimeError("invalid operator in temporal formula: {}".format(str_location(x.location)))
                while not unary and self.__check(operator):
                    self.__reduce()
                self.__stack.append((operator, unary))
                unary = True
            self.__stack.append(element.term)
            unary = False
        while len(self.__stack) > 1:
            self.__reduce()
        return self.__stack[0]

def parse_raw_formula(x):
    """
    Turns the given unparsed term into a term.
    """
    return TheoryParser().parse(x)

# {{{1 theory_term -> term

class TheoryTermToTermTransformer(Transformer):
    """
    This class transforms a given theory term into a plain term.
    """
    def visit_TheoryTermSequence(self, x):
        """
        Theory term tuples are mapped to term tuples.
        """
        if x.sequence_type == _ast.TheorySequenceType.Tuple:
            return _ast.Function(x.location, "", [self(a) for a in x.arguments], False)
        else:
            raise RuntimeError("invalid term: {}".format(str_location(x.location)))

    def visit_TheoryFunction(self, x):
        """
        Theory functions are mapped to functions.
        """
        isnum = lambda y: y.type == _ast.ASTType.Symbol and y.symbol.type == _clingo.SymbolType.Number
        if x.name == "-" and len(x.arguments) == 1:
            rhs = self(x.arguments[0])
            if isnum(rhs):
                return _ast.Symbol(x.location, _clingo.Number(-rhs.symbol.number))
            else:
                return _ast.UnaryOperation(x.location, _ast.UnaryOperator.Minus, rhs)
        elif (x.name == "+" or x.name == "-") and len(x.arguments) == 2:
            lhs = self(x.arguments[0])
            rhs = self(x.arguments[1])
            op  = _ast.BinaryOperator.Plus if x.name == "+" else _ast.BinaryOperator.Minus
            if isnum(lhs) and isnum(rhs):
                lhs = lhs.symbol.number
                rhs = rhs.symbol.number
                return _ast.Symbol(x.location, _clingo.Number(lhs + rhs if x.name == "+" else lhs - rhs))
            else:
                return _ast.BinaryOperation(x.location, op, lhs, rhs)
        elif x.name == "-" and len(x.arguments) == 2:
            return _ast.BinaryOperation(x.location, _ast.BinaryOperator.Minus, self(x.arguments[0]), self(x.arguments[1]))
        elif (x.name, TheoryParser.binary) in TheoryParser.table or (x.name, TheoryParser.unary) in TheoryParser.table:
            raise RuntimeError("operator not handled: {}".format(str_location(x.location)))
        else:
            return _ast.Function(x.location, x.name, [self(a) for a in x.arguments], False)

    def visit_TheoryUnparsedTerm(self, x):
        """
        Unparsed term are first parsed and then handled by the transformer.
        """
        return self.visit(parse_raw_formula(x))

def theory_term_to_term(x):
    """
    Convert the given theory term into a term.
    """
    return TheoryTermToTermTransformer()(x)

# {{{1 transform

class ProgramTransformer(Transformer):
    """
    Rewrites all parity constraints in a program.

    Members:
    __add    -- Callback to add auxiliary statements.
    __remove -- Boolean indicating that the next rule should be removed.
    """
    def __init__(self, add):
        self.__add    = add
        self.__remove = False
        self.__id     = 0

    def visit_Rule(self, rule):
        """
        Returns the result of rewriting a rule.
        """
        rule = self.visit_children(rule)
        if self.__remove:
            self.__remove = False
            return None
        else:
            return rule

    def visit_TheoryAtom(self, atom):
        """
        Rewrites theory atoms related to parity constraints.
        """
        if atom.term.type == _ast.ASTType.Function and len(atom.term.arguments) == 0:
            if atom.term.name in ["odd", "even"]:
                self.__remove = True
                i = _ast.Symbol(atom.location, _clingo.Number(self.__id))
                ct = _ast.Symbol(atom.location, _clingo.Function(atom.term.name))
                head = _ast.SymbolicAtom(_ast.Function(atom.location, g_aux_name, [i, ct], False))
                head = _ast.Literal(atom.location, _ast.Sign.NoSign, head)
                self.__add(_ast.Rule(atom.location, head, []))
                for element in atom.elements:
                    head = _ast.Function(atom.location, "", [theory_term_to_term(t) for t in element.tuple], False)
                    head = _ast.SymbolicAtom(_ast.Function(atom.location, g_aux_name, [i, ct, head], False))
                    head = _ast.Literal(atom.location, _ast.Sign.NoSign, head)
                    body = element.condition
                    self.__add(_ast.Rule(atom.location, head, body))
                self.__id += 1
        return atom

def transform(inputs, add,xors):
    """
    Rewrites the given statement if it is a parity constraint and the resulting
    rules to the program using a callback.

    Arguments:
    statement -- The statement to rewrite.
    add       -- Callback to add statements to the logic program beeing
                 rewritten.
    """


    pt = ProgramTransformer(add)
    def add_if_not_none(statement):
        statement = pt(statement)
        if statement is not None:
            add(statement)

    '''for s in inputs:
        _clingo.parse_program(s, add_if_not_none)
    _clingo.parse_program(xors,add_if_not_none)'''

    _clingo.parse_program(inputs, add_if_not_none)
    _clingo.parse_program(xors,add_if_not_none)

