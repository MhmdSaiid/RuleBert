
from src.lpmln_parser import lpmln_parser
from src import marginal_mcsat
import time

class learn_general_ga_mcasp(lpmln_parser):

    def __init__(self,asp_content,evid_content,xorMode,max_learning_iteration=50,max_mcasp_sample=50,lr=0.1,complete_evide=False, min_mcasp_sample=1,stopping_diff=0.001,ini_weight=1):
        self.aspProgram = asp_content
        self.evidence =evid_content
        self.xorMode = xorMode
        self.lr = lr
        self.max_learning_iteration = max_learning_iteration
        self.min_mcsat_interation = min_mcasp_sample
        self.max_mcsat_iteration = max_mcasp_sample
        self.stopping_diff = stopping_diff
        self.init_weight = ini_weight

        self. progranWithoutPlaceholder = ""
        self.weightsDic = {}
        self.args=[]
        self.complete_evid=complete_evide


    def findTotalMisWithMCSAT(self,finalout):
        for key, value in self.weightsDic.items():
            value['unsatEvi'] = 0
            value['unsatNoEvi'] = 0

        if isinstance(self.evidence,list):
            for i,evid in enumerate(self.evidence):

                if self.args.verbosity > 4:
                    print("Start sample no evidenced")

                mcASPObj = marginal_mcsat.mcSAT(finalout, "", [], self.xorMode, self.max_mcsat_iteration)
                mcASPObj.args = self.args
                mcASPObj.runMCASP()
                samples = mcASPObj.sampleForReturn
                if self.args.verbosity > 4:
                    print("Done sample no evidenced")
                for samplene in samples:
                    for atom in samplene:
                        counter = 0
                        if (atom.name == "neg" or atom.name == "unsat") and ("lpmlnneg_" in str(atom.arguments[0])):
                            idx = eval(str(atom.arguments[0]).split('_')[1])
                            counter += 1
                            for key, value in self.weightsDic.items():
                                if value['wIndex'] == idx:
                                    value['unsatNoEvi'] += 1


                if self.args.verbosity > 4:
                    if self.complete_evid:
                        print("Start sampling from "+str(i)+"th complete evidenced")
                    else:
                        print("Start sampling from "+str(i)+"th partial evidenced")

                if self.complete_evid:
                    mcASPObj = marginal_mcsat.mcSAT(finalout, evid, [], self.xorMode, 1)
                else:
                    mcASPObj = marginal_mcsat.mcSAT(finalout, evid, [], self.xorMode, self.max_mcsat_iteration)

                mcASPObj.args = self.args
                mcASPObj.runMCASP()
                samplesEvidenced = mcASPObj.sampleForReturn

                if self.args.verbosity > 4:
                    if self.complete_evid:
                        print("Done sampling from "+str(i)+"th complete evidenced")
                    else:
                        print("Done sampling from "+str(i)+"th partial evidenced")
                for sampleE in samplesEvidenced:
                    for atom in sampleE:
                        if (str(atom.name) == "neg" or str(atom.name) == "unsat") and "lpmlnneg_" in str(
                                atom.arguments[0]):
                            idx = eval(str(atom.arguments[0]).split('_')[1])
                            for key, value in self.weightsDic.items():
                                if value['wIndex'] == idx:
                                    if self.complete_evid:
                                        value['unsatEvi'] += 1 * self.max_mcsat_iteration
                                    else:
                                        value['unsatEvi'] += 1
        else:
            if self.args.verbosity > 4:
                if self.complete_evid:
                    print("Start sampling from complete evidenced")
                else:
                    print("Start sampling from partial evidenced")
            if self.complete_evid:
                mcASPObj = marginal_mcsat.mcSAT(finalout,self.evidence, [], self.xorMode, 1)
            else:
                mcASPObj = marginal_mcsat.mcSAT(finalout,self.evidence, [], self.xorMode, self.max_mcsat_iteration)
            mcASPObj.args = self.args
            mcASPObj.runMCASP()
            samplesEvidenced = mcASPObj.sampleForReturn
            if self.args.verbosity > 4:
                if self.complete_evid:
                    print("Done sample from complete evidenced")
                else:
                    print("Done sample from partial evidenced")

            for sampleE in samplesEvidenced:
                for atom in sampleE:
                    if (str(atom.name) == "neg" or str(atom.name) == "unsat") and "lpmlnneg_" in str(atom.arguments[0]):
                        idx = eval(str(atom.arguments[0]).split('_')[1])
                        for key, value in self.weightsDic.items():
                            if value['wIndex'] == idx:
                                if self.complete_evid:
                                    value['unsatEvi'] += 1 * self.max_mcsat_iteration
                                else:
                                    value['unsatEvi'] += 1


        if self.args.verbosity > 4:
            print(self.weightsDic)

    def updateWithWeightPlaceHolders(self):
        self.progranWithoutPlaceholder = ""
        buffer = []
        rule_index = 1
        for line in self.aspProgram.splitlines():
            if line.startswith('@w'):
                buffer.append(str(self.weightsDic[str(rule_index)]['weight']) + ' ' + ''.join(line.split(' ')[1:]))
            else:
                buffer.append(line)
            if "".join(line.split()) != "":
                rule_index += 1

        for line in buffer:
            self.progranWithoutPlaceholder += line + "\n"

    def updatefinalWeight(self):
        buffer = []
        rule_index = 1
        for line in self.aspProgram.splitlines():
            if line.startswith('@w'):
                buffer.append(str(self.weightsDic[str(rule_index)]['weightSum'] / self.max_learning_iteration) + ' ' + ''.join(
                    line.split(' ')[1:]))
            else:
                buffer.append(line)
            if "".join(line.split()) != "":
                rule_index += 1
        clingo_opt = open("lpmln_learned.weight", 'w')
        for line in buffer:
            clingo_opt.write(line+'\n')
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
                    'unsatEvi': 0,
                    'unsatNoEvi': 0,
                    'weightSum': 0,
                }
                self.weightsDic[str(rule_index)]['weight'] += self.init_weight
                buffer.append(str(self.weightsDic[str(rule_index)]['weight']) + ' ' + ''.join(line.split(' ')[1:]))
            else:
                buffer.append(line)
            if "".join(line.split()) != "":
                rule_index += 1
        for line in buffer:
            self.progranWithoutPlaceholder += line + "\n"




    def learn(self):
        self.learn_ini()
        content = self.lpmln_to_lpmln_neg_parser(self.progranWithoutPlaceholder)
        content = self.lpmln_to_asp_parser(content)
        finalout = self.asp_domain_2_asp_parser(content)


        if isinstance(self.evidence,list):
            for l in self.evidence:
                mcASPObj = marginal_mcsat.mcSAT(finalout, l, [], self.xorMode, self.min_mcsat_interation)
                mcASPObj.args = self.args
                result = mcASPObj.runMCASP()
                if not result:
                    print('Evidence and program not satisfiable. Exit.')
                    return False
        else:
            mcASPObj = marginal_mcsat.mcSAT(finalout, l, [], self.xorMode, self.min_mcsat_interation)
            mcASPObj.args = self.args
            result = mcASPObj.runMCASP()
            if not result:
                print('Evidence and program not satisfiable. Exit.')
                return False

        # Begin: Learning Iterations
        actualNumIteration = 0
        for iter_count in range(self.max_learning_iteration):
            iter_start_time = time.time()
            actualNumIteration += 1

            self.updateWithWeightPlaceHolders()
            content = self.lpmln_to_lpmln_neg_parser(self.progranWithoutPlaceholder)
            content = self.lpmln_to_asp_parser(content)
            finalout = self.asp_domain_2_asp_parser(content)


            self.findTotalMisWithMCSAT(finalout)
            # End: Single learning iteration
            # Compute new weights
            total_gradient = 0

            max_diff = 0

            for key, value in self.weightsDic.items():
                prob_gradient = float(-value['unsatEvi'] + float(value['unsatNoEvi']))/float(self.max_mcsat_iteration)
                total_gradient += abs(prob_gradient)

                if self.args.verbosity > 4:
                    print('Rule: ', key)
                    print('# False ground instances from Evidence: ', float(value['unsatEvi']) / float(self.max_mcsat_iteration))
                    print('Expected # false ground instances: ', float(value['unsatNoEvi']) / float(self.max_mcsat_iteration))
                    print('Gradient: ', prob_gradient)




                if max_diff < abs(self.lr * prob_gradient):
                    max_diff = abs(self.lr * prob_gradient)
                value['weight'] += (self.lr * prob_gradient)
                value['weightSum'] += value['weight']
                if self.args.verbosity > 4:
                    print('New weight: ', key, ':', value['weight'])
            # End: Learning Iterations

            if self.args.verbosity > 4:
                print("max_diff: ", max_diff)


            if max_diff <= self.stopping_diff:
                break

            if self.args.verbosity > 4:
                print('============ Time for finishing iteration ' + str(actualNumIteration) + ': '+str(time.time()-iter_start_time)+' ============')

        # Begin: Store and save new weights
        self.updateWithWeightPlaceHolders()
        self.updatefinalWeight()
        # End: Store and save new weights
