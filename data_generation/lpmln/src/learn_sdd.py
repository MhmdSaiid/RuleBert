from src import sdd_infer
import clingo
import math
from src import lpmln_parser
from src import run


class learn_sdd(object):

    def __init__(self,asp_content,observation_content,save=None,load=None,lr=0.1,max_learn_iteration = 1000,stopping_diff=0.0001,ini_weight=1):
        self.aspProgram = asp_content
        self.observation = observation_content
        self.progranWithoutPlaceholder = ""
        self.lr = lr
        self.stopping_diff = stopping_diff
        self.init_weight = ini_weight
        self.weightsDic = {}
        self.sddName = ""
        self.learnedIteration = 1
        self.max_learn_iteration = max_learn_iteration
        self.load= load
        self.save = save
        self.args=[]

    def updateSDDMappingDicWeight(self,sddInferObj):
        for key,value in sddInferObj.nameMappingDic.items():
            symbol = clingo.parse_term(key)
            if str(symbol.type) == "Function":
                for key_1, value_1 in self.weightsDic.items():
                    if (eval(str(symbol.arguments[0])) == value_1["wIndex"]) and symbol.name=="unsat":
                        tempList = symbol.arguments
                        tempList[1] =  str(value_1["weight"])
                        newSymbol = clingo.Function(symbol.name,tempList)
                        sddInferObj.nameMappingDic[str(newSymbol)] = sddInferObj.nameMappingDic.pop(key)



    def updatefinalWeight(self):
        buffer = []
        rule_index = 1
        for line in self.aspProgram.splitlines():
            if line.startswith('@w'):
                buffer.append(str(self.weightsDic[str(rule_index)]['weightSum'] / self.learnedIteration) + ' ' + ''.join(
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


    def wmc_with_weight(self,sddInferObj):
        for key, value in self.weightsDic.items():
            value['gradient'] = 0.0
        for key,value in self.weightsDic.items():
            for name,node in sddInferObj.nameMappingDic.items():
                symbol = clingo.parse_term(name)
                if str(symbol.type) == "Function" and symbol.name=="unsat":
                    if str(key) == str(symbol.arguments[0]):

                        weight = eval(str(symbol.arguments[1]).strip("\""))
                        wmc, weight_org = sddInferObj.learn_iniWMC()

                        d_exp = math.exp(weight)

                        d_org_weight = wmc.literal_derivative(-eval(str(node))) * d_exp

                        if isinstance(self.observation,list):
                            for i,ob in enumerate(self.observation):

                                if self.args.verbosity > 4:
                                    print("Leraning from Observation: ", i)

                                wmc, weight_observation = sddInferObj.learn_ini_WMC_with_observation(sddInferObj.parseEvidence(ob))
                                d_weight_observation = wmc.literal_derivative(-eval(str(node))) * d_exp
                                d = (d_weight_observation / weight_observation - d_org_weight / weight_org)
                                value['gradient']+=d
                        else:
                            wmc, weight_observation = sddInferObj.learn_ini_WMC_with_observation(sddInferObj.parseEvidence(self.observation))

                            d_weight_observation = wmc.literal_derivative(-eval(str(node))) * d_exp

                            d = (d_weight_observation / weight_observation - d_org_weight / weight_org)
                            value['gradient'] += d



    '''
    1. Take out the placeholder "@w", and replace it be the initial weight value.
    2. Generate weight table for all rules needs to be learned. 
    3. Generate SDD for \Pi
    '''
    def learn_ini(self, sdd_construction_method):
        buffer = []
        rule_index = 1
        for line in self.aspProgram.splitlines():
            if line.startswith('@w'):
                self.weightsDic[str(rule_index)] = {
                    'wIndex': rule_index,
                    'weight': 0.0,
                    'gradient':0.0,
                    'weightSum': 0.0,
                }
                self.weightsDic[str(rule_index)]['weight'] += self.init_weight
                buffer.append(str(self.weightsDic[str(rule_index)]['weight']) + ' ' + ''.join(line.split(' ')[1:]))
            else:
                buffer.append(line)
            if "".join(line.split()) != "":
                rule_index += 1
        for line in buffer:
            self.progranWithoutPlaceholder += line + "\n"

        parser = lpmln_parser.lpmln_parser()
        content = parser.lpmln_to_asp_sdd_parser(self.progranWithoutPlaceholder)
        finalout = parser.asp_domain_2_asp_parser(content)

        sdd_obj = sdd_infer.sddInference("", [])
        sdd_obj.args = self.args

        if sdd_construction_method == 0:

            if self.load is None:
                if self.save is None:
                    sdd_obj.sddConstructorFromLPMLN(finalout)
                else:
                    sdd_obj.sddConstructorFromLPMLN(finalout, True,self.save)
            else:
                sdd_obj.sddConstructorFromFile(self.load)

            return sdd_obj
        elif sdd_construction_method ==1:
            if self.load is None:
                r = run.run(finalout, self.args)
                if self.save is None:
                    sdd_f_name = r.get_info_4_learn()
                    sdd_obj.sddConstructorFromFile(sdd_f_name)

                    import os
                    if os.path.exists(sdd_f_name+".map") and os.path.exists(sdd_f_name+".vtree") and os.path.exists(sdd_f_name+".sdd"):
                        os.remove(sdd_f_name+".map")
                        os.remove(sdd_f_name+".sdd")
                        os.remove(sdd_f_name+".vtree")
                else:
                    sdd_obj.sddConstructorFromFile(r.get_info_4_learn())
            else:
                sdd_obj.sddConstructorFromFile(self.load)

            return sdd_obj










    def learn(self,sdd_construction_method=0):
        sdd_learn_obj = self.learn_ini(sdd_construction_method)

        max_diff = float("inf")
        while max_diff > self.stopping_diff and self.learnedIteration< self.max_learn_iteration:
            self.learnedIteration+=1
            self.wmc_with_weight(sdd_learn_obj)
            for key,value in self.weightsDic.items():
                prob_gradient = value['gradient']
                if max_diff > abs(self.lr * prob_gradient):
                    max_diff = abs(self.lr * prob_gradient)
                value['weight'] += (self.lr * prob_gradient)
                value['weightSum'] += value['weight']
            if self.args.verbosity > 4:
                print("max_diff: ", max_diff)
            self.updateSDDMappingDicWeight(sdd_learn_obj)

            if self.args.verbosity > 4:
                print('============ Iteration ' + str(self.learnedIteration) + ' ============')
                print(max_diff)
                for key,value in self.weightsDic.items():
                    print(key,value["weightSum"]/self.learnedIteration)
        # Begin: Store and save new weights
        self.updateSDDMappingDicWeight(sdd_learn_obj)
        self.updatefinalWeight()
        # End: Store and save new weights








