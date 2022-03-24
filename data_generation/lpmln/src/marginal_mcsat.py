#script(python)
import clingo
import os
import random
import sympy
import time
from src import xor_constraint_drawer

class mcSAT(object):

    def __init__(self,content,evidence,queryList,xorMode,max_liter=500):
        warn_option = "--warn=none"
        thread_option = "-t 4"
        self.clingoOptions = [warn_option, thread_option]
        self.max_num_iteration = max_liter
        self.curr_sample = None
        self.whole_model = []
        self.query_count = {}
        self.domain = []
        self.aspContent = content
        self.eviContent = evidence
        self.queryList = queryList
        self.sampleForReturn = []
        self.xorM = xorMode
        self.args=[]



    def findUnsatRules(self,atoms):
        M = []
        for atom in atoms:
            if atom.name.startswith('unsat'):
                weight = float(str(atom.arguments[1]).replace("\"", ""))
                r = random.random()
                if r < 1 - sympy.exp(weight):
                    M.append(atom)
        return M

    def processSample(self,atoms):
        # Find rules that are not satisfied
        M = self.findUnsatRules(atoms)
        # Do specific things with the sample: counting atom occurence
        self.sampleForReturn.append(atoms)

        for a in atoms:
            if a in self.query_count:
                self.query_count[a] += 1

        return M


    def runMCASP(self):
        # Configure Clingo running options
        firstSamplcontrol = clingo.Control(self.clingoOptions)
        firstSamplcontrol.add("base",[],self.aspContent)
        firstSamplcontrol.ground([("base", [])])

        if self.eviContent != "":
            firstSamplcontrol.add("evid",[],self.eviContent)
            firstSamplcontrol.ground([("evid", [])])

        for atom in firstSamplcontrol.symbolic_atoms:
            if atom.symbol.name in self.queryList:
                self.query_count[atom.symbol] = 0
        random.seed()
        sample_count = 0
        models = []
        firstSamplcontrol.solve([], lambda model: models.append(model.symbols(atoms=True)))
        if len(models) >= 1:
            # randomly generate a index from models
            randomIndex = random.randint(0, len(models) - 1)
            model = models[randomIndex]
        else:
            print("Program has no satisfiable solution, exit!")
            firstSamplcontrol.cleanup()
            return False


        M = self.processSample(model)
        startT = time.time()

        deduct = 0

        for _ in range(1, self.max_num_iteration):

            if self.args.verbosity>0 and sample_count % 10 == 0 and sample_count !=0 :
                print("Got number of samples: ", sample_count)
                endT = time.time()
                print("time = ",endT-startT)
                startT = time.time()
            elif self.args.verbosity>4:
                print("Got number of samples: ", sample_count)
                endT = time.time()
                print("time = ", endT - startT)
                startT = time.time()

            sample_count += 1
            # Create file with satisfaction constraints

            constraintContent = ""
            for m in M:
                argsStr = ''
                for arg in m.arguments:
                    argsStr += (str(arg) + ',')
                argsStr = argsStr.rstrip(',')
                constraintContent+=':- not ' + m.name + '(' + argsStr + ').\n'
            #startTime = time.time()

            if self.eviContent != "":
                xorSampler = xor_constraint_drawer.xorSampler(self.xorM,[self.aspContent, self.eviContent, constraintContent],self.clingoOptions,None,None,deduct)
                xorSampler.args = self.args
                models = xorSampler.startDrawSample()
                if deduct>0:
                    deduct = xorSampler.deduct -1
                else:
                    deduct = xorSampler.deduct
            else:
                xorSampler = xor_constraint_drawer.xorSampler(self.xorM,[self.aspContent, constraintContent], self.clingoOptions,None,None,deduct)
                xorSampler.args = self.args
                models = xorSampler.startDrawSample()
                if deduct>0:
                    deduct = xorSampler.deduct -1
                else:
                    deduct = xorSampler.deduct
            #print("MCASP time for getting 1 sample: ", str(time.time() - startTime))
            if len(models) > 1:
                # randomly generate a index from models
                randomIndex = random.randint(0, len(models) - 1)
                model = models[randomIndex]
            else:
                model = models[0]
            M = self.processSample(model)


        firstSamplcontrol.cleanup()
        return True

    def printQuery(self):
        for atom in self.query_count:
            print(atom, ": ", float(self.query_count[atom]) / float(self.max_num_iteration))
            #print(self.query_count[atom])
            #print(self.max_num_iteration)

    def getSamples(self):
        return self.sampleForReturn



