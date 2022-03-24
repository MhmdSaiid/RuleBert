from src import logic
from collections import OrderedDict


def simplify(prog):
    new_rules = []
    for rule in prog.rules:
        new_rule = logic.Rule(simplify_formula(rule.head, prog.symbolTable),
                              simplify_formula(rule.body, prog.symbolTable))
        new_rules.append(new_rule)
    new_constraint = simplify_formula(prog.constraint, prog.symbolTable)
    new_parameters = []
    for param in prog.parameters:
        new_param = simplify_formula(param, prog.symbolTable)
        new_parameters.append(new_param)
    new_weights = OrderedDict()
    for lit, weight in prog.weights.item():
        newlit = simplify_formula(lit, prog.symbolTable)
        new_weights[newlit] = weight
    new_program = logic.LogicProgram(new_rules, new_constraint, new_parameters, prog.symbolTable, new_weights)
    return new_program


def simplify_formula(formula, symbol_table):
    # Flyweights doen iets raar met types. type(formula) is ground.LIterals werkt bijvoorbeeld niet.
    # Daarom de lelijke hacks
    if formula is logic.TrueForm():
        return formula
    elif formula is logic.FalseForm():
        return formula
    elif isinstance(formula, type(logic.Literal("", True))):
        return logic.Literal(symbol_table.get(formula.atomnumber, formula.atomnumber), formula.negated)
    elif type(formula) is logic.Disjunction:
        new_subforms = []
        for subform in formula.subforms:
            new_subform = simplify_formula(subform, symbol_table)
            new_subforms.append(new_subform)
        result = logic.Disjunction(new_subforms)
        return result
    elif type(formula) is logic.Conjunction:
        new_subforms = []
        for subform in formula.subforms:
            new_subform = simplify_formula(subform, symbol_table)
            new_subforms.append(new_subform)
        result = logic.Conjunction(new_subforms)
        return result
    else:
        print(type(formula))
        raise NotImplementedError