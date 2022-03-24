import os.path
import re
import numpy as np
import clingo
import time
import sys
from src import lpmln_parser
class MVPP(object):
    def __init__(self, program,args, k=1, eps=0.000001):
        self.k = k
        self.eps = eps

        # each element in self.pc is a list of atoms (one list for one prob choice rule)
        self.pc = []
        # each element in self.parameters is a list of probabilities
        self.parameters = []
        # each element in self.learnable is a list of Boolean values
        self.learnable = []
        # self.asp is the ASP part of the LPMLN program
        self.asp = ""
        # self.pi_prime is the ASP program \Pi' defined for the semantics
        self.pi_prime = ""
        # self.remain_probs is a list of probs, each denotes a remaining prob given those non-learnable probs
        self.remain_probs = []
        self.pc, self.parameters, self.learnable, self.asp, self.pi_prime, self.remain_probs = self.parse(program)
        self.normalize_probs()
        self.args=args


        if self.args.verbosity>=5:
            print("=============PI_PRIME===============")
            print(self.pi_prime)

    def parse(self, program):
        pc = []
        parameters = []
        learnable = []
        asp = ""
        pi_prime = ""
        remain_probs = []

        lines = []
        # if program is a file
        if os.path.isfile(program):
            with open(program, 'r') as program:
                lines = program.readlines()
                # print("lines1: {}".format(lines))
        # if program is a string containing all rules of an LPMLN program
        elif type(program) is str and program.strip().endswith("."):
            lines = program.split('\n')
            # print("lines2: {}".format(lines))
        else:
            print("Error! The MVPP program is not valid.")
            sys.exit()

        for line in lines:
            if re.match(r".*[0-1]\.?[0-9]*\s.*;.*", line):
                list_of_atoms = []
                list_of_probs = []
                list_of_bools = []
                choices = line.strip()[:-1].split(";")
                for choice in choices:
                    splited = choice.strip().split()
                    prob = splited[0]
                    atom=''.join(splited[1:])

                    list_of_atoms.append(atom.replace(" ",""))
                    if prob.startswith("@"):
                        list_of_probs.append(float(prob[1:]))
                        list_of_bools.append(True)
                    else:
                        list_of_probs.append(float(prob))
                        list_of_bools.append(False)
                pc.append(list_of_atoms)
                parameters.append(list_of_probs)
                learnable.append(list_of_bools)
                pi_prime += "1{"+"; ".join(list_of_atoms)+"}1.\n"
            else:
                asp += (line.strip()+"\n")

        pi_prime =  asp +pi_prime

        for ruleIdx, list_of_bools in enumerate(learnable):
            remain_prob = 1
            for atomIdx, b in enumerate(list_of_bools):
                if b == False:
                    remain_prob -= parameters[ruleIdx][atomIdx]
            remain_probs.append(remain_prob)


        paser = lpmln_parser.lpmln_parser()


        return pc, parameters, learnable, asp, paser.asp_domain_2_asp_parser(pi_prime), remain_probs

    def normalize_probs(self):
        for ruleIdx, list_of_bools in enumerate(self.learnable):
            summation = 0
            # 1st, we turn each probability into [0,1]
            for atomIdx, b in enumerate(list_of_bools):
                if b == True:
                    if self.parameters[ruleIdx][atomIdx] >=1 :
                        self.parameters[ruleIdx][atomIdx] = 1- self.eps
                    elif self.parameters[ruleIdx][atomIdx] <=0:
                        self.parameters[ruleIdx][atomIdx] = self.eps

            # 2nd, we normalize the probabilities
            for atomIdx, b in enumerate(list_of_bools):
                if b == True:
                    summation += self.parameters[ruleIdx][atomIdx]
            for atomIdx, b in enumerate(list_of_bools):
                if b == True:
                    self.parameters[ruleIdx][atomIdx] = self.parameters[ruleIdx][atomIdx] / summation * self.remain_probs[ruleIdx]

        return True

    def prob_of_interpretation(self, I):
        prob = 1.0
        while not isinstance(I[0], str):
            I = I[0]
        for ruleIdx,list_of_atoms in enumerate(self.pc):
            for atomIdx, atom in enumerate(list_of_atoms):
                if atom in I:
                    prob = prob * self.parameters[ruleIdx][atomIdx]
        return prob

    # we assume obs is a string containing a valid Clingo program, 
    # and each obs is written in constraint form
    def find_one_SM_under_obs(self, obs):
        program = self.pi_prime + obs
        # print("program:\n{}\n".format(program))
        clingo_control = clingo.Control(["--warn=none"])
        models = []
        # print("\nPi': \n{}".format(program))
        clingo_control.add("base", [], program)
        # print("point 3")
        clingo_control.ground([("base", [])])
        # print("point 4")
        clingo_control.solve(None, lambda model: models.append(model.symbols(atoms=True)))
        # print("point 5")
        models = [[str(atom) for atom in model] for model in models]
        # print("point 6")
        # print("All stable models of Pi' under obs \"{}\" :\n{}\n".format(obs,models))
        return models

    # we assume obs is a string containing a valid Clingo program, 
    # and each obs is written in constraint form
    def find_all_SM_under_obs(self, obs):
        program = self.pi_prime + obs
        # print("program:\n{}\n".format(program))
        clingo_control = clingo.Control(["0", "--warn=none"])
        models = []
        # print("\nPi': \n{}".format(program))
        clingo_control.add("base", [], program)
        # print("point 3")
        clingo_control.ground([("base", [])])
        # print("point 4")
        clingo_control.solve(None, lambda model: models.append(model.symbols(atoms=True)))
        # print("point 5")
        models = [[str(atom) for atom in model] for model in models]
        # print("point 6")
        # print("All stable models of Pi' under obs \"{}\" :\n{}\n".format(obs,models))
        return models

    # compute P(O)
    def inference_obs_exact(self, obs):
        prob = 0
        models = self.find_all_SM_under_obs(obs)
        for I in models:
            prob += self.prob_of_interpretation(I)
        return prob

    def gradient(self, ruleIdx, atomIdx,obs_index):
        # we will compute P(I)/p_i where I satisfies obs and c=v_i
        p_obs_i = 0
        # we will compute P(I)/p_j where I satisfies obs and c=v_j for i!=j
        p_obs_j = 0
        # we will compute P(I) where I satisfies obs
        p_obs = 0

        # 1st, we generate all I that satisfies obs
        #models = self.find_all_SM_under_obs(obs)
        models = self.all_SM_under_obs[obs_index]
        # 2nd, we iterate over each model I, and check if I satisfies c=v_i
        c_equal_vi = self.pc[ruleIdx][atomIdx]
        p_i = self.parameters[ruleIdx][atomIdx]
        for I in models:
            p_I = self.prob_of_interpretation(I)
            # print("I: {}\t p_I: {}\t p_i: {}".format(I,p_I,p_i))
            p_obs += p_I
            if c_equal_vi in I:
                # if p_i == 0:
                #     p_i = self.eps
                p_obs_i += p_I/p_i
            else:
                for atomIdx2, p_j in enumerate(self.parameters[ruleIdx]):
                    c_equal_vj = self.pc[ruleIdx][atomIdx2]
                    if c_equal_vj in I:
                        # if p_j == 0:
                        #     p_j = self.eps
                        p_obs_j += p_I/p_j

        # 3rd, we compute gradient
        # print("p_obs_i: {}\t p_obs_j: {}\t p_obs: {}".format(p_obs_i,p_obs_j,p_obs))
        # if p_obs == 0:
        #     p_obs = self.eps
        gradient = (p_obs_i-p_obs_j)/p_obs

        return gradient

    # gradients are stored in numpy array instead of list
    # obs is a string
    def gradients_one_obs(self, obs_index):
        gradients = [[0.0 for item in l] for l in self.parameters]
        for ruleIdx,list_of_bools in enumerate(self.learnable):
            for atomIdx, b in enumerate(list_of_bools):
                if b == True:
                    # print("ruleIdx: {}\t atomIdx: {}\t obs: {}".format(ruleIdx, atomIdx, obs))
                    gradients[ruleIdx][atomIdx] = self.gradient(ruleIdx, atomIdx, obs_index)
        return np.array(gradients)

    '''# gradients are stored in numpy array instead of list
    def gradients_multi_obs(self, list_of_obs):
        gradients = np.array([[0.0 for item in l] for l in self.parameters])
        for obs in list_of_obs:
            gradients += self.gradients_one_obs(obs)
        return gradients'''


    ##############################
    ####### Sampling Method ######
    ##############################

    # it will generate k sample stable models for a k-coherent program under a specific total choice
    def k_sample(self):
        asp_with_facts = self.asp
        clingo_control = clingo.Control(["0", "--warn=none"])
        models = []
        for ruleIdx,list_of_atoms in enumerate(self.pc):
            tmp = np.random.choice(list_of_atoms, 1, p=self.parameters[ruleIdx])
            # print(tmp)
            asp_with_facts += tmp[0]+".\n"
        clingo_control.add("base", [], asp_with_facts)
        clingo_control.ground([("base", [])])
        result = clingo_control.solve(None, lambda model: models.append(model.symbols(shown=True)))
        models = [[str(atom) for atom in model] for model in models]
        # print("k")
        # print(models)
        return models

    # it will generate k*num sample stable models
    def sample(self, num=1):
        models = []
        for i in range(num):
            models = models + self.k_sample()
        # print("test")
        # print(models)
        return models

    # it will generate at least num of samples that satisfy obs
    def sample_obs(self, obs, num=50):
        count = 0
        models = []
        while count < num:
            asp_with_facts = self.asp
            asp_with_facts += obs
            clingo_control = clingo.Control(["0", "--warn=none"])
            models_tmp = []
            for ruleIdx,list_of_atoms in enumerate(self.pc):
                # print("parameters before: {}".format(self.parameters[ruleIdx]))
                # self.normalize_probs()
                # print("parameters after: {}\n".format(self.parameters[ruleIdx]))
                tmp = np.random.choice(list_of_atoms, 1, p=self.parameters[ruleIdx])
                # print(tmp)
                asp_with_facts += tmp[0]+".\n"


            clingo_control.add("base", [], asp_with_facts)
            clingo_control.ground([("base", [])])
            result = clingo_control.solve(None, lambda model: models_tmp.append(model.symbols(shown=True)))
            if str(result) == "SAT":
                models_tmp = [[str(atom) for atom in model] for model in models_tmp]
                # print("models_tmp:")
                # print(models_tmp)
                count += len(models_tmp)
                models = models + models_tmp
                # print("count: {}".format(count))
            elif str(result) == "UNSAT":
                pass
            else:
                print("Error! The result of a clingo call is not SAT nor UNSAT!")
        return models

    # we compute the gradients (numpy array) w.r.t. all probs in the ruleIdx-th rule
    # given models that satisfy obs
    def gradient_given_models(self, ruleIdx, models):
        arity = len(self.parameters[ruleIdx])

        # we will compute N(O) and N(O,c=v_i)/p_i for each i
        n_O = 0
        n_i = [0]*arity

        # 1st, we compute N(O)
        n_O = len(models)

        # 2nd, we compute N(O,c=v_i)/p_i for each i
        for model in models:
            for atomIdx, atom in enumerate(self.pc[ruleIdx]):
                if atom in model:
                    n_i[atomIdx] += 1
        for atomIdx, p_i in enumerate(self.parameters[ruleIdx]):
            # if p_i == 0:
            #     p_i = self.eps
            n_i[atomIdx] = n_i[atomIdx]/p_i
        
        # 3rd, we compute the derivative of L'(O) w.r.t. p_i for each i
        tmp = np.array(n_i) * (-1)
        summation = np.sum(tmp)
        gradients = np.array([summation]*arity)
        # print(summation)
        # gradients = np.array([[summation for item in l] for l in self.parameters])
        # print("init gradients: {}".format(gradients))
        for atomIdx, p_i in enumerate(self.parameters[ruleIdx]):
            gradients[atomIdx] = gradients[atomIdx] + 2* n_i[atomIdx]
        gradients = gradients / n_O
        # print("n_O: {}".format(n_O))
        # print("n_i: {}\t n_O: {}\t gradients: {}".format(n_i, n_O, gradients))
        return gradients


    # gradients are stored in numpy array instead of list
    # obs is a string
    def gradients_one_obs_by_sampling(self, obs, num=50):
        gradients = np.array([[0.0 for item in l] for l in self.parameters])
        # 1st, we generate at least num of stable models that satisfy obs
        models = self.sample_obs(obs=obs, num=num)

        # 2nd, we compute the gradients w.r.t. the probs in each rule
        for ruleIdx,list_of_bools in enumerate(self.learnable):
            gradients[ruleIdx] = self.gradient_given_models(ruleIdx, models)
            for atomIdx, b in enumerate(list_of_bools):
                if b == False:
                    gradients[ruleIdx][atomIdx] = 0
        # print(gradients)
        return gradients

    # we compute the gradients (numpy array) w.r.t. all probs given list_of_obs
    def gradients_multi_obs_by_sampling(self, list_of_obs, num=50):
        gradients = np.array([[0.0 for item in l] for l in self.parameters])

        # we itereate over all obs
        for obs in list_of_obs:
            # 1st, we generate at least num of stable models that satisfy obs
            models = self.sample_obs(obs=obs, num=num)

            # 2nd, we accumulate the gradients w.r.t. the probs in each rule
            for ruleIdx,list_of_bools in enumerate(self.learnable):
                gradients[ruleIdx] += self.gradient_given_models(ruleIdx, models)
                for atomIdx, b in enumerate(list_of_bools):
                    if b == False:
                        gradients[ruleIdx][atomIdx] = 0
        # print(gradients)
        return gradients

    # we compute the gradients (numpy array) w.r.t. all probs given list_of_obs
    # while we generate at least one sample without considering probability distribution
    def gradients_multi_obs_by_one_sample(self, list_of_obs):
        gradients = np.array([[0.0 for item in l] for l in self.parameters])

        # we itereate over all obs
        for obs in list_of_obs:
            # 1st, we generate one stable model that satisfy obs
            models = self.find_one_SM_under_obs(obs=obs)

            # 2nd, we accumulate the gradients w.r.t. the probs in each rule
            for ruleIdx,list_of_bools in enumerate(self.learnable):
                gradients[ruleIdx] += self.gradient_given_models(ruleIdx, models)
                for atomIdx, b in enumerate(list_of_bools):
                    if b == False:
                        gradients[ruleIdx][atomIdx] = 0
        # print(gradients)
        return gradients




    def learn(self,list_of_obs,num_of_samples=50, lr=0.01, thres=0.0001, max_iter=None, num_pretrain=5):
        time_init = time.time()
        check_continue = True
        # Step 1: Parameter Pre-training: we pretrain the parameters
        # so that it's easier to generate sample stable models
        assert type(num_pretrain) is int
        if num_pretrain >= 1:
            if self.args.verbosity>=6:
                print("\n#######################################################\nParameter Pre-training for {} iterations...\n#######################################################".format(num_pretrain))
            for iteration in range(num_pretrain):
                if self.args.verbosity >= 6:
                    print("\n#### Iteration {} for Pre-Training ####\nGenerating 1 stable model for each observation...\n".format(iteration + 1))
                dif = lr * self.gradients_multi_obs_by_one_sample(list_of_obs)
                self.parameters = (np.array(self.parameters) + dif).tolist()
                self.normalize_probs()
                if self.args.verbosity >= 6:
                    print("After {} seconds of training (in total)".format(time.time() - time_init))
                    print("Current parameters: {}".format(self.parameters))


        # For each observation, decide which observation uses learn exact or learn sample
        learn_obs_decider = []
        for obs in list_of_obs:
            crl = clingo.Control(['0',"--warn=none"])
            crl.add("base", [], self.pi_prime+obs)
            crl.ground([("base", [])])
            with crl.solve(yield_=True) as handle:
                counter = 0
                for m in handle:
                    counter += 1
                    if counter==2500:
                        break
                if counter ==2500:
                    learn_obs_decider.append('sample')
                else:
                    learn_obs_decider.append('exact')

        if self.args.verbosity >= 6:
            print("Leraning from Observations Decider: ",learn_obs_decider)


        self.all_SM_under_obs = {}
        for i,method in enumerate(learn_obs_decider):
            if method == 'exact':
                self.all_SM_under_obs[i] =self.find_all_SM_under_obs(list_of_obs[i])

        iteration = 1

        while check_continue:
            if self.args.verbosity >= 4:
                print("\n#### Iteration {} ####".format(iteration))
            old_parameters = np.array(self.parameters)
            check_continue = False

            total_dif = np.array([[0.0 for item in l] for l in self.parameters])

            for i,obs in enumerate(list_of_obs):
                if learn_obs_decider[i] == 'sample':
                    dif = lr * self.gradients_multi_obs_by_sampling([obs], num=num_of_samples)
                else:
                    dif = lr * self.gradients_one_obs(i)
                    for ruleIdx, list_of_bools in enumerate(self.learnable):
                        # 1st, we turn each gradient into [-0.2, 0.2]
                        for atomIdx, b in enumerate(list_of_bools):
                            if b == True:
                                if dif[ruleIdx][atomIdx] > 0.2:
                                    dif[ruleIdx][atomIdx] = 0.2
                                elif dif[ruleIdx][atomIdx] < -0.2:
                                    dif[ruleIdx][atomIdx] = -0.2
                total_dif+=dif

            self.parameters = (np.array(self.parameters) + total_dif).tolist()
            self.normalize_probs()
            if self.args.verbosity >= 6:
                print("After {} seconds of training (in total)".format(time.time() - time_init))
                print("Current parameters: {}".format(self.parameters))

            # we termintate if the change of the parameters is lower than thres
            dif = np.array(self.parameters) - old_parameters
            dif = abs(max(dif.min(), dif.max(), key=abs))
            if self.args.verbosity >= 6:
                print("Max change on probabilities: {}".format(dif))

            iteration += 1
            if dif > thres:
                check_continue = True
            if max_iter is not None:
                if iteration > max_iter:
                    check_continue = False

        if self.args.verbosity >= 4:
            print("Final parameters: {}".format(self.parameters))
        self.write_final_program()

    def write_final_program(self):

        soft_content = ""

        for prob,pc in zip(self.parameters,self.pc):
            for a,b in zip(prob,pc):
                soft_content += str(a)+ ' ' + str(b) + ";"
            soft_content=soft_content[:-1]
            soft_content+='.\n'

        with open("lpmln_learned.weight",'w') as fw:
            fw.write(soft_content+self.asp)

        if self.args.verbosity>=5:
            print("################# Final Program: #####################")
            print(soft_content+self.asp)






