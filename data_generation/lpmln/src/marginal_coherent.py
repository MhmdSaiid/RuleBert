#script(python)
import clingo
import random
import math
import time
class Marginal_coherent(object):

    def __init__(self,content,evidence,queryList,max_liter=500):
        warn_option = "--warn=none"
        thread_option = "-t 4"
        self.clingoOptions = [warn_option, thread_option]
        self.max_num_iteration = max_liter
        self.curr_sample = None
        self.args=[]

        self.content = content
        self.queryList = queryList
        self.query_count = {}
        self.gnd_atoms = []
        firstSamplcontrol = clingo.Control(self.clingoOptions)
        firstSamplcontrol.add("base", [], self.content)
        firstSamplcontrol.ground([("base", [])])

        self.condition_mode = False

        if evidence != "":
            self.process_evidence(evidence)
            self.condition_mode = True
            self.num_satisfy_evid = 0

        for atom in firstSamplcontrol.symbolic_atoms:
            self.gnd_atoms.append(atom.symbol)
            if atom.symbol.name in self.queryList:
                self.query_count[atom.symbol] = 0
    def get_c(self):
        constraints_2_return = ""
        for atom in self.gnd_atoms:
            if atom.name == "unsat":
                weight = eval(str(atom.arguments[1]).replace("\"", ""))

                if 1- (math.exp(weight)/(1+math.exp(weight))) > random.random():
                    constraints_2_return+= ":- not "+str(atom) + ".\n"
                else:
                    constraints_2_return+= ":- "+str(atom) + ".\n"
        return constraints_2_return


    def process_models(self,models):
        if self.condition_mode:
            for model in models:
                satisfied = True
                for evidence_atom in self.evidence_atom_assignemnt:
                    if (evidence_atom[1] and evidence_atom[0] in model) or (not evidence_atom[1] and evidence_atom[0] not in model):
                        pass
                    else:
                        satisfied=False
                        break
                if satisfied:
                    self.num_satisfy_evid +=1
                else:
                    continue
                for atom in model:
                    if atom in self.query_count.keys():
                        self.query_count[atom]+=1
        else:
            for model in models:
                for atom in model:
                    if atom in self.query_count.keys():
                        self.query_count[atom]+=1

    def process_evidence(self,raw_evidence):
        evidSet = raw_evidence.split('.')
        self.evidence_atom_assignemnt = []
        for e in evidSet:
            e = e.strip('\n')
            if e == "":
                continue
            elif "not" in e:
                self.evidence_atom_assignemnt.append((clingo.parse_term((e.split("not")[1].strip(" "))), True))
            else:
                self.evidence_atom_assignemnt.append((clingo.parse_term((e.split(":-")[1].strip(" "))), False))


    def run(self):
        sample_count = 0
        startT = time.time()
        for _ in range(1, self.max_num_iteration):

            if self.args.verbosity > 0 and sample_count % 10 == 0 and sample_count != 0:
                print("Got number of samples: ", sample_count)
                endT = time.time()
                print("time = ", endT - startT)
                startT = time.time()
            elif self.args.verbosity > 4:
                print("Got number of samples: ", sample_count)
                endT = time.time()
                print("time = ", endT - startT)
                startT = time.time()
            sample_count += 1

            solver_control = clingo.Control(self.clingoOptions)
            solver_control.add("base", [], self.content)
            solver_control.ground([("base", [])])


            solver_control.add("c_cons",[],self.get_c())
            solver_control.ground([("c_cons", [])])

            models = []
            solver_control.solve([], lambda model: models.append(model.symbols(atoms=True)))
            self.process_models(models)
            solver_control.cleanup()





    def printQuery(self):
        for atom in self.query_count:
            print(atom, ": ", float(self.query_count[atom]) / float(self.max_num_iteration))
            #print(self.query_count[atom])
            #print(self.max_num_iteration)





