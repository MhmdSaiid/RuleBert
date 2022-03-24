from src import processor
import clingo
from src import xor_constraint_drawer
import random
import sympy


class learn_simple_comp_evid(object):
    def __init__(self,asp_content,evid_content,max_learning_iteration=50,lr=0.1,stopping_diff = 0.000000000000001,init_weight = 1):

        self.aspProgram = asp_content
        self.evidence = evid_content

        self.lr = lr
        self.max_learning_iteration = max_learning_iteration
        self.stopping_diff = stopping_diff
        self.init_weight = init_weight
        self.progranWithoutPlaceholder = ""
        self.weightsDic = {}

        warn_option = "--warn=none"
        thread_option = "-t 4"
        self.clingoOptions = [warn_option, thread_option]
        self.args = []

    def updatefinalWeight(self):
        buffer = []
        rule_index = 1
        for line in self.aspProgram.splitlines():
            if line.startswith('@w'):
                buffer.append(str(self.weightsDic[str(rule_index)]['weight']) + ' ' + ''.join(line.split(' ')[1:]))
            else:
                buffer.append(line)
            if "".join(line.split()) != "":
                rule_index += 1
        clingo_opt = open("lpmln_learned.weight", 'w')
        for line in buffer:
            clingo_opt.write(line + '\n')
        clingo_opt.close()

        if self.args.verbosity>4:
            print("############################ Learned Program ####################")
            for line in buffer:
                print(line)





    def learn_ini(self):

        buffer = []
        rule_index = 1
        for line in self.aspProgram.splitlines():
            if line.startswith('@w'):
                self.weightsDic[str(rule_index)] = {
                    'wIndex': rule_index,
                    'weight': 0,
                    'n': 0,
                    'nii': 0,
                    'atomName': str(line.split(' ')[1]).split('(')[0],
                    'gradient': [],
                }
                self.weightsDic[str(rule_index)]['weight'] += self.init_weight
                buffer.append(str(self.weightsDic[str(rule_index)]['weight']) + ' ' + ''.join(line.split(' ')[1:]))
            else:
                buffer.append(line)
            if "".join(line.split()) != "":
                rule_index += 1

        for line in buffer:
            self.progranWithoutPlaceholder += line + "\n"

        # Done on creating weight file, and initializing dictionary

        content = processor.lpmln_to_asp_parser(self.progranWithoutPlaceholder)
        finalout = processor.asp_domain_2_asp_parser(content)

        warn_option = "--warn=none"
        thread_option = "-t 4"
        options = [warn_option, thread_option]
        sampling_control = clingo.Control(options)
        sampling_control.add("base",[],finalout)
        sampling_control.ground([("base", [])])
        symbols = [atom.symbol for atom in sampling_control.symbolic_atoms]


        if isinstance(self.evidence,list):
            for i,evid in enumerate(self.evidence):
                for s in symbols:
                    for key, value in self.weightsDic.items():
                        if s.name == value['atomName']:
                            value['n'] += 1

                evidenced_program_control = clingo.Control(options)
                evidenced_program_control.add("base", [], finalout+evid)
                evidenced_program_control.ground([("base", [])])

                models = []
                solve_result = evidenced_program_control.solve([], lambda model: models.append(model.symbols(atoms=True)))

                if not solve_result:
                    print("Program is unsatifiable. EXIT")

                if len(models) > 1:
                    # randomly generate a index from models
                    randomIndex = random.randint(0, len(models) - 1)
                    model = models[randomIndex]
                else:
                    model = models[0]

                for atom in model:
                    if str(atom.name) == "unsat":
                        idx = eval(str(atom.arguments[0]))
                        for key, value in self.weightsDic.items():
                            if value['wIndex'] == idx:
                                value['nii'] += 1
        else:
            for s in symbols:
                for key, value in self.weightsDic.items():
                    if s.name == value['atomName']:
                        value['n'] += 1
            evidenced_program_control = clingo.Control(options)
            evidenced_program_control.add("base", [], finalout + self.evidence)
            evidenced_program_control.ground([("base", [])])

            models = []
            solve_result = evidenced_program_control.solve([], lambda model: models.append(model.symbols(atoms=True)))

            if not solve_result:
                print("Program is unsatifiable. EXIT")

            if len(models) > 1:
                # randomly generate a index from models
                randomIndex = random.randint(0, len(models) - 1)
                model = models[randomIndex]
            else:
                model = models[0]

            for atom in model:
                if str(atom.name) == "unsat":
                    idx = eval(str(atom.arguments[0]))
                    for key, value in self.weightsDic.items():
                        if value['wIndex'] == idx:
                            value['nii'] += 1


    def learn(self):
        self.learn_ini()
        # Begin: Learning Iterations

        max_diff = float("inf")
        iteration = 0
        while iteration< self.max_learning_iteration :
            iteration+=1
            if self.args.verbosity > 4:
                print('============ Iteration ' + str(iteration) + ' ============')
            for key, value in self.weightsDic.items():
                if self.args.verbosity>4:
                    print(value['nii'], value['n'] / (1 + sympy.exp(float(value['weight']))))

                gradient = value['n'] / (1 + sympy.exp(float(value['weight']))) - value['nii']
                value['gradient'].append(self.lr * gradient)
                value['weight'] += value['gradient'][-1]


                if max_diff > abs(value['gradient'][-1]):
                    max_diff = abs(value['gradient'][-1])
            if self.args.verbosity>4:
                print("max_diff: ", max_diff)
            if max_diff <= self.stopping_diff:
                break

            # End: Learning Iterations

        # Begin: Store and save new weights

        self.updatefinalWeight()

        # End: Store and save new weights
