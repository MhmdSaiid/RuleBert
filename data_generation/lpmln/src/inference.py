import math
from src import compile
from lib import sdd as sddm
import math
import src.parse_tool as parser
#  ====================
#  Interface
#  ====================


def do_inference(compiled_programs,query,evidence,):
    return Inference(compiled_programs,query,evidence).inferece()

def count_models(compiled_programs):
    return Inference(compiled_programs,[],[]).count_models()

class Inference(object):
    def __init__(self,compiled_programs,query_list,evidence):
        self.compiled_programs = compiled_programs
        self.query_list = query_list
        self.evidence = evidence
        self.program_weightDict,self.evidence_weightDict,self.query_weightDict = self._fill_weight_tables()


    def _fill_weight_tables(self):
        two_val_compiled_prog = self.compiled_programs[0]
        program_weightDict = {}
        evidence_weightDict = {}
        query_weightDict = {}

        evidences_set = parser.parse_evidence(self.evidence)

        for key, item in two_val_compiled_prog.program.symbolTable.items():
            symbol = parser.parse(item)
            if symbol.name == "unsat":
                for atom,sdd_num in two_val_compiled_prog.atom2sddnum.items():
                    if key == atom.atomnumber:
                        program_weightDict[-sdd_num] = eval(str(symbol.arguments[1]).strip('\"'))

            if symbol.name in self.query_list:
                for atom,sdd_num in two_val_compiled_prog.atom2sddnum.items():
                    if key == atom.atomnumber:
                        query_weightDict[symbol] = -sdd_num
            for e in evidences_set:
                if e.hard and e.symbol.to_string() == symbol.to_string():
                    for atom, sdd_num in two_val_compiled_prog.atom2sddnum.items():
                        if key == atom.atomnumber:
                            evidence_weightDict[e.assignment * sdd_num] = 0

        return program_weightDict,evidence_weightDict,query_weightDict


    def count_models(self):
        two_val_compiled_prog = self.compiled_programs[0]
        sdd = two_val_compiled_prog.get_entire_program_as_sdd()
        manager = two_val_compiled_prog.manager

        print("MODEL COUNT: ")
        print(compile.compute_model_count(sdd, manager))


    def inferece(self):
        manager = self.compiled_programs[0].manager
        sdd = self.compiled_programs[0].get_entire_program_as_sdd()
        infer_result = []
        for key,item in self.query_weightDict.items():
            w_after = compile.compute_weighted_model_count(sdd,manager,self.program_weightDict,self.evidence_weightDict,item)
            w_before = compile.compute_weighted_model_count(sdd,manager,self.program_weightDict,self.evidence_weightDict)

            w_after = math.exp(w_after)
            w_before = math.exp(w_before)
            infer_result.append([key,w_after/w_before])
        return infer_result



    def learn(self):

        manager = self.compiled_programs[0].manager
        sdd = self.compiled_programs[0].get_entire_program_as_sdd()
        infer_result = []
        for key, item in self.query_weightDict.items():
            w_after = compile.compute_weighted_model_count(sdd, manager, self.program_weightDict,
                                                           self.evidence_weightDict, item)
            w_before = compile.compute_weighted_model_count(sdd, manager, self.program_weightDict,
                                                            self.evidence_weightDict)

            w_after = math.exp(w_after)
            w_before = math.exp(w_before)
            infer_result.append([key, w_after / w_before])



        return infer_result



























































'''def count_models_projected(compiled_programs):
    """
    Counts the number of models of a compiled program projected onto the vocabulary specified by the symbolTable
    :param compiled_programs: list of compiled programs. This method assumes only one program in the list
    :return: the number of projected models of this program
    """
    compile_result = compiled_programs[0]
    vocabulary = compile_result.program.get_outvoc()
    sdd = compile_result.get_entire_program_as_sdd_over_voc(vocabulary)
    manager = compile_result.manager
    model_count = compile.compute_model_count(sdd, manager)
    # The SDD library does not return the exact model count but the model count of the projection onto the set of
    # variables that occur in the SDD. To get the exact model count, we need to multiply by 2 for each unused variable.
    # However, we also need to be careful since only unused variables that occur in the symbol table need to be taken
    #  into account. Therefor, we count the used vars and compare it the size of the symbol table
    usedvars = sddm.sdd_variables(sdd, manager)
    # NOTE index zero of the array usedvars is never used!
    nbvars_in_manager = sddm.sdd_manager_var_count(manager)
    nbusedvars = 0
    for i in range(1, nbvars_in_manager + 1):
        if sddm.sdd_array_int_element(usedvars, i) == 1:
            nbusedvars += 1
    nbunusedvars = len(vocabulary) - nbusedvars
    model_count *= int(math.pow(2, nbunusedvars))
    print("PROJECTED MODEL COUNT: ")
    print(model_count)'''


