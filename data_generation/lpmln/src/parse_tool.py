import clingo
class Symbol(object):
    def __init__(self,name,arguments):
        self.name = name
        self.arguments = arguments

    def to_string(self):
        if self.arguments==[]:
            return self.name
        else:
            return self.name + "("+(",".join(self.arguments))+")"



class Evidence(object):
    def __init__(self,symbol,assignment,hard):
        self.symbol = symbol
        self.assignment = assignment
        self.hard = hard

def parse(in_symbol):
    if '(' in in_symbol and ')' in in_symbol:
        name,rest = in_symbol.split('(')
        rest = rest[:-1]
        arguments = rest.split(',')
        return Symbol(name,arguments)
    else:
        return Symbol(in_symbol,[])


def parse_evidence(evidence):
    evidences_set = []
    if evidence != "" and isinstance(evidence, str):
        for line in evidence.split('.'):
            if ":-" in line and "not" in line:
                evidences_set.append(Evidence(parse((line.split("not")[1].strip(" "))), -1, True))  # True atom, set -1 * atom to 0
            elif ":-" in line and not "not" in line:
                evidences_set.append(Evidence(parse((line.split(":-")[1].strip(" "))), 1, True)) # False atom, set 1*atom to 0
            else:
                evidences_set.append(Evidence(parse(line[:-1]), None, False))
    elif isinstance(evidence, list):
        for e in evidence:
            if isinstance(e[0], Symbol):
                sym = e[0]
            else:
                tempList = []
                if len(e[0].arguments) != 0:
                    for element in e[0].arguments:
                        tempList.append(str(element))
                sym = Symbol(e[0].name, tempList)

            if e[1]:
                evidences_set.append(Evidence(sym,-1,True))
            else:
                evidences_set.append(Evidence(sym,1,True))
    return evidences_set





















