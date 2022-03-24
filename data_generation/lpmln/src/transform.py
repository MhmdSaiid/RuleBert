from src.logic import *


def transform(program):
    """
    Performs the Gelfond-Lifschitz reduct transformation on a logic program.
    This transformation introduces one new variable for each atom occuring negated in the body of this program.
    These new variables are added as parameters, and constraitns that they should equal the original value are added
    in the constraint
    Besides the transformed program, a mapping from variables to their internal copy is returned as well
    :param program: the program to transform
    :return:  1, the transformation of the program 2, a mapping from variables to their copy
    """
    rules = []
    constraint = program.constraint
    parameters = list(program.parameters)



    varmap = {}
    for rule in program.rules:
        head = rule.head
        orig_body = rule.body
        new_body, constraint = _replace_neg_lits_in_formula(orig_body, constraint, parameters, varmap)
        rules.append(Rule(head, new_body))

    outprogram = LogicProgram(rules, constraint, parameters, program.symbolTable, program.weights)
    return outprogram, varmap


# =========
# INTERNALS
# =========

def _replace_neg_lits_in_formula(formula, constraint, parameters, varmap):
    """
    Replaces all negative literals in the formula by an internal literal called lp2sdd_i_name,
    where name is the name of the original literal
    Furthermore, adds this new literal to the set of parameters (if not yet in there) and adds a constraint to the
    theory that this literal is equivalent to the original
    :param formula: a formula to replace the literals in
    :param constraint: the constraint to which we add the equivalence demands
    :param parameters: the set of parameters
    :param varmap: a mapping from variables to their copy
    :return: both a new formula and the updated constraint
    """
    if formula.is_true():
        return TrueForm(), constraint
    elif formula.is_false():
        return FalseForm(), constraint
    elif isinstance(formula, type(Literal("", True))):
        if formula.negated:
            return _get_literal(formula, constraint, parameters, varmap)
        else:
            return formula, constraint
    elif type(formula) is Disjunction:
        new_subforms = []
        for subform in formula.subforms:
            new_subform, constraint = _replace_neg_lits_in_formula(subform, constraint, parameters, varmap)
            new_subforms.append(new_subform)
        result = Disjunction(new_subforms)
        return result, constraint
    elif type(formula) is Conjunction:
        new_subforms = []
        for subform in formula.subforms:
            new_subform, constraint = _replace_neg_lits_in_formula(subform, constraint, parameters, varmap)
            new_subforms.append(new_subform)
        result = Conjunction(new_subforms)
        return result, constraint
    else:
        raise NotImplementedError("Replacing negative literals in formula of type " + str(type(formula)))


def _get_literal(lit, constraint, parameters, varmap):
    """
    returns an internal literal called lp2sdd_i_name where name is the name of lit.
    Furthermore, adds this new literal to the set of parameters (if not yet in there) and adds a constraint to the
    theory that this literal is equivalent to the original
    :param lit: the literal to create a copy of
    :param constraint: the constraint to add the equivalence to
    :param parameters: the set of parameters
    :return: both the new set of literals and the updated constraint
    """
    result = Literal("lp2sdd_i_"+str(lit.atomnumber), lit.negated)
    new_atom = result
    orig_atom = lit
    if new_atom.negated:
        new_atom = new_atom.negate()
        orig_atom = orig_atom.negate()
    varmap[orig_atom] = new_atom
    if new_atom not in parameters:
        parameters.append(new_atom)
        equiv = new_atom.equivalent(orig_atom)
        # TODO REMOVED SINCE THIS ENFORCED ANYWAY constraint = constraint.conjoin(equiv)
    return result, constraint