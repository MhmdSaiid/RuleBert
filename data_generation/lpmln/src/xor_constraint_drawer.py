import random
import math
import clingo

import time
import subprocess
import os
import sys

import lib.xorro.tester as xt


fn_xorro = os.path.join(os.path.dirname(__file__), 'xorro')
sys.path.append('fn_xorro')


class xorSampler:

    def __init__(self,xormode,programs,clingoOptions,s=-1,q=-1,deduct=0):
        self.deduct = deduct
        self.xormode = xormode
        self.prog=""
        for p in programs:
            self.prog+=p
        self.clingoOptions = clingoOptions
        self.s = s
        self.q = q
        self.args = []

    def getConstrains(self,sampling_control):
        symbols = [atom.symbol for atom in sampling_control.symbolic_atoms if atom.is_fact == False]
        if self.s ==-1:
            self.s = int(math.log(len(symbols) + 1, 2))-self.deduct
        else:
            self.s = self.s-self.deduct
        if self.q == -1:
            self.q = 0.3

        constraints = []
        parities = []
        for i in range(self.s):
            size = random.randint(int(round(len(symbols) * self.q)), int(round(len(symbols) * self.q)))
            constraint_size = (size * 100 / len(symbols))
            if constraint_size == 0:
                size = 1
            # Ramdon draw a set of size number of atoms from "symbols" pool, and append the set to "constraints set"
            constraints.append(random.sample(symbols, size))
            # Random select the parity for this literation of constraint
            parities.append(random.randint(0, 1))

        for i in range(len(constraints)):
            for j in range(len(constraints[i])):
                terms = "%s:%s" % (constraints[i][j], constraints[i][j])
                constraints[i][j] = terms

        integr_constr = ""
        for index in range(len(constraints)):
            terms = " ; ".join(str(x) for x in constraints[index])
            integr_constr += ":- N = #count{ %s }, N\\2!=%s. \n" % (terms, parities[index])
        return integr_constr


    def drawSample(self):
        #Create clingo object for sampling(only for grounding purpose)
        sampling_control = clingo.Control(self.clingoOptions)
        models = []
        models_before = []

        sampling_control.add("base",[],self.prog)
        sampling_control.ground([("base", [])])
        while True:
            xorConstrannsToBeAdded = self.getConstrains(sampling_control)
            sampling_control.add("constraint", [], xorConstrannsToBeAdded)
            sampling_control.ground([("constraint", [])])
            solve_result = sampling_control.solve([], lambda model: models.append(model.symbols(atoms=True)))

            if solve_result.satisfiable:
                models_before = models
                models = []
                if xorConstrannsToBeAdded == "":
                    return models_before
                else:
                    continue
            else:
                return models_before


    def startDrawSample(self):
        if self.xormode == 2:
            #-------------------------------------------Using NEW XORRO Implementation START-------------------------------------------

            empty_sample_counter=0

            while True:
                empty_sample_counter+=1
                if empty_sample_counter%2==0:
                    self.deduct+=1
                    if self.args.verbosity>6:
                        print(">>>>> Increament deduct. Current deduct is: ",self.deduct)

                models = xt.Application(self.args).main(clingo.Control(self.clingoOptions), self.prog,self.deduct)
                if models!=None:
                    return models
                elif self.args.verbosity>6:
                    print(">>>>> Get None models, re-sampling...")




        # -------------------------------------------Using NEW XORRO Implementation START-------------------------------------------
        else:
            #-------------------------------------------Using OLD Implementation START-------------------------------------------

            test_control = clingo.Control(self.clingoOptions)
            test_control.add("base",[],self.prog)
            test_control.ground([("base", [])])
            solve_result = test_control.solve()
            models = []
            if solve_result.satisfiable:
                if self.xormode == 0:
                    while len(models) == 0:
                        for _ in range(10): # Try 10 times at every deduct level.
                            models = self.drawSample()
                            if len(models)!=0:
                                return models
                        self.deduct += 1
                elif self.xormode == 1:
                    while len(models) == 0:
                        models = self.drawSample()
            return models
            #-------------------------------------------Using OLD Implementation OLD-------------------------------------------



