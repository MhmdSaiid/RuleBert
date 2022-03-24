import clingo
import random
import sympy
import time
import math
from sympy import *
import itertools
from src import sdd_infer
from src import run
from src import exactProbInfer
from src import marginal_mcsat

class decision_cal:
    def __init__(self,aspContent,decisionPredicates,xormode=2,max_sample=50,p=0.5,mt=5,mf=10):
        warn_option = "--warn=none"
        thread_option = "-t 4"
        self.clingoOptions = [warn_option, thread_option]
        self.max_sample = max_sample
        self.mt=mt
        self.mf=mf
        self.p = p
        self.aspContent = aspContent
        self.decisionPredicates = decisionPredicates

        self.grounded_dec_atoms = []
        self.soln = ""
        self.utility = -float("inf")
        self.xorMode = xormode
        self.args = []

        firstSamplcontrol = clingo.Control(self.clingoOptions)
        firstSamplcontrol.add("base", [], self.aspContent)
        firstSamplcontrol.ground([("base", [])])

        for atom in firstSamplcontrol.symbolic_atoms:
            if atom.symbol.name in self.decisionPredicates:
                self.grounded_dec_atoms.append(atom.symbol)


    def expectedUtility(self,query_action):
        mcASPObj = marginal_mcsat.mcSAT(self.aspContent, query_action, [], self.xorMode,self.max_sample)
        mcASPObj.args = self.args

        mcASPObj.runMCASP()
        samplesEvidenced = mcASPObj.getSamples()
        #Given program combining with decision or evidence cannot lead to any stable model
        if len(samplesEvidenced) == 0:
            return False
        expectedU = 0.0
        for sampleE in samplesEvidenced:
            for atom in sampleE:
                if str(atom.name) == "utility":
                    expectedU+=eval(str(atom.arguments[0]).strip('\"'))
        return expectedU/len(samplesEvidenced)

    def expectedExactUtility(self,query_action):
        content = self.aspContent+ query_action
        opt1 = self.clingoOptions[0]
        opt2 = self.clingoOptions[1]
        opt3 = "--opt-mode=enum"
        opt4 = "0"
        info = exactProbInfer.exactProbInfer(content,[opt1,opt2,opt3,opt4],[""],False,False)
        info.solve()
        info.reset()
        return info.totalUtility


    def expectedExactUtility_SDD(self,query_action,save=None,load=None):
        content = self.aspContent
        inferObj = sdd_infer.sddInference(query_action, ['utility'])
        inferObj.args = self.args

        if save is None and load is None:
            inferObj.sddConstructorFromLPMLN(content)
        elif save is None and load is not None:
            inferObj.sddConstructorFromFile(load)
        else:
            inferObj.sddConstructorFromLPMLN(content,True,save)

        result = inferObj.inference(inferObj.parseEvidence())
        total = 0
        for atom in result:
            reward = eval(str(atom[0].arguments[0]).strip('\"'))
            total += reward * atom[1]
        return "Utility: " + str(total)


    def expectedExactUtility_SDD_2(self,query_action,save=None,load=None):
        r = run.run(self.aspContent, self.args)
        r.args = self.args
        result = r.infer(['utility'],query_action)
        total = 0
        for atom in result:
            reward = eval(str(atom[0].arguments[0]).strip('\"'))
            total += reward * atom[1]
        return "Utility: " + str(total)




    def satisfiableChecker(self,utilityAssignemnt):
        checkerControl = clingo.Control(self.clingoOptions)
        checkerControl.add("base", [], self.aspContent)
        checkerControl.ground([("base", [])])

        checkerControl.add("utilityAssign", [], utilityAssignemnt)
        checkerControl.ground([("utilityAssign", [])])

        solve_result = checkerControl.solve()
        return solve_result.satisfiable




    def buildASPEvidenceForm(self,decisionAssignments):
        decisionEvidence = ""

        for decision in decisionAssignments:
            if decision[1]:
                decisionEvidence+= ':- not ' + str(decision[0]) + '.\n'
            else:
                decisionEvidence+= ':- ' + str(decision[0]) + '.\n'
        return decisionEvidence


    def flipDecisionAtom(self,decisionAssignments,decisionToBeFlipped):
        flippedAssignments = []

        for decision in decisionAssignments:
            if str(decision[0]) == str(decisionToBeFlipped[0]):
                flipDecision = (decision[0], not decision[1])
                flippedAssignments.append(flipDecision)
            else:
                flippedAssignments.append(decision)
        return flippedAssignments


    def completeTruthAssignment(self,trueAtoms):
        decisionAssignments = []
        for gda in self.grounded_dec_atoms:
            if gda in trueAtoms:
                decisionAssignments.append((gda,True))
            else:
                decisionAssignments.append((gda,False))
        return decisionAssignments



    def max_walk_sat(self):
        solution = ""
        utility = -float("inf")
        for mt_iter in range(1,self.mt):
            if self.args.verbosity >4:
                print("Starting one iteration: ",mt_iter)

            mcASPObj = marginal_mcsat.mcSAT(self.aspContent, "", [], self.xorMode,1)
            mcASPObj.runMCASP()
            firstSample = mcASPObj.getSamples()
            decisionAssignments = self.completeTruthAssignment(firstSample[0])

            tempSoln = decisionAssignments
            tempUtility = self.expectedUtility(self.buildASPEvidenceForm(tempSoln))

            for _ in range(1,self.mf):
                vf=""
                deltCostVf = float("inf")
                orgUtility = self.expectedUtility(self.buildASPEvidenceForm(tempSoln))

                if self.p < random.random():
                    vf = tempSoln[random.randint(0, len(tempSoln) - 1)]
                    newDecisionAssignments = self.flipDecisionAtom(tempSoln,vf)
                    flippedUtility = self.expectedUtility(self.buildASPEvidenceForm(newDecisionAssignments))
                    if flippedUtility != False:
                        deltCostVf = orgUtility - flippedUtility
                else:
                    deltaCost = []
                    for decToFlip in tempSoln:
                        newDecisionAssignments = self.flipDecisionAtom(tempSoln, decToFlip)
                        flippedUtility = self.expectedUtility(self.buildASPEvidenceForm(newDecisionAssignments))
                        if flippedUtility != False:
                            deltaCost.append((decToFlip,orgUtility - flippedUtility))
                    if len(deltaCost) != 0:
                        minDeltaCost = min(deltaCost,key=lambda t:t[1])
                    else:
                        continue

                    vf = minDeltaCost[0]
                    deltCostVf = minDeltaCost[1]

                if deltCostVf<0:
                    tempSoln = self.flipDecisionAtom(tempSoln, vf)
                    tempUtility = tempUtility - deltCostVf

            if tempUtility >utility:
                utility = tempUtility
                solution = tempSoln

        return solution,utility


    def max_walk_sat_exact(self):
        solution = ""
        utility = -float("inf")
        for _ in range(1,self.mt):
            mcASPObj = marginal_mcsat.mcSAT(self.aspContent, "", [], self.xorMode,1)
            mcASPObj.runMCASP()
            firstSample = mcASPObj.getSamples()
            decisionAssignments = self.completeTruthAssignment(firstSample[0])

            tempSoln = decisionAssignments
            tempUtility = self.expectedExactUtility(self.buildASPEvidenceForm(tempSoln))

            for _ in range(1,self.mf):
                vf=""
                deltCostVf = float("inf")
                orgUtility = self.expectedExactUtility(self.buildASPEvidenceForm(tempSoln))

                if self.p < random.random():
                    vf = tempSoln[random.randint(0, len(tempSoln) - 1)]
                    newDecisionAssignments = self.flipDecisionAtom(tempSoln,vf)
                    flippedUtility = self.expectedExactUtility(self.buildASPEvidenceForm(newDecisionAssignments))
                    if flippedUtility != False:
                        deltCostVf = orgUtility - flippedUtility
                else:
                    deltaCost = []
                    for decToFlip in tempSoln:
                        newDecisionAssignments = self.flipDecisionAtom(tempSoln, decToFlip)
                        flippedUtility = self.expectedExactUtility(self.buildASPEvidenceForm(newDecisionAssignments))
                        if flippedUtility != False:
                            deltaCost.append((decToFlip,orgUtility - flippedUtility))
                    if len(deltaCost) != 0:
                        minDeltaCost = min(deltaCost,key=lambda t:t[1])
                    else:
                        continue

                    vf = minDeltaCost[0]
                    deltCostVf = minDeltaCost[1]

                if deltCostVf<0:
                    tempSoln = self.flipDecisionAtom(tempSoln, vf)
                    tempUtility = tempUtility - deltCostVf

            if tempUtility >utility:
                utility = tempUtility
                solution = tempSoln

        return solution,utility



    def max_walk_sat_exact_helper(self,infer_obj,dec_assignment):
        infer_obj.queryAtoms = []
        result = infer_obj.inference(dec_assignment)

        total = 0
        for atom in result:
            reward = eval(str(atom[0].arguments[0]).strip('\"'))
            total += reward * atom[1]
        return total


    def max_walk_sat_exact_sdd(self,save=None,load=None):
        solution = ""
        utility = -float("inf")
        inferObj = sdd_infer.sddInference("", ['utility'])
        inferObj.args = self.args
        if load is None and save is None:
            inferObj.sddConstructorFromLPMLN(self.aspContent)
        elif load is None and save is not None:
            inferObj.sddConstructorFromLPMLN(self.aspContent,True,save)
        else:
            inferObj.sddConstructorFromFile(load)

        for _ in range(1,self.mt):
            mcASPObj = marginal_mcsat.mcSAT(self.aspContent, "", [], self.xorMode,1)
            mcASPObj.runMCASP()
            firstSample = mcASPObj.getSamples()
            decisionAssignments = self.completeTruthAssignment(firstSample[0])

            tempSoln = decisionAssignments
            tempUtility = self.max_walk_sat_exact_helper(inferObj,tempSoln)

            for _ in range(1,self.mf):
                vf=""
                deltCostVf = float("inf")
                orgUtility = self.max_walk_sat_exact_helper(inferObj,tempSoln)

                if self.p < random.random():
                    vf = tempSoln[random.randint(0, len(tempSoln) - 1)]
                    newDecisionAssignments = self.flipDecisionAtom(tempSoln,vf)
                    flippedUtility =self.max_walk_sat_exact_helper(inferObj,newDecisionAssignments)
                    if flippedUtility != False:
                        deltCostVf = orgUtility - flippedUtility
                else:
                    deltaCost = []
                    for decToFlip in tempSoln:
                        newDecisionAssignments = self.flipDecisionAtom(tempSoln, decToFlip)
                        flippedUtility = self.max_walk_sat_exact_helper(inferObj,newDecisionAssignments)
                        if flippedUtility != False:
                            deltaCost.append((decToFlip,orgUtility - flippedUtility))
                    if len(deltaCost) != 0:
                        minDeltaCost = min(deltaCost,key=lambda t:t[1])
                    else:
                        continue

                    vf = minDeltaCost[0]
                    deltCostVf = minDeltaCost[1]

                if deltCostVf<0:
                    tempSoln = self.flipDecisionAtom(tempSoln, vf)
                    tempUtility = tempUtility - deltCostVf

            if tempUtility >utility:
                utility = tempUtility
                solution = tempSoln

        return solution,utility



    ########################################Not well developed, fix it later ###############################

    def max_walk_sat_sdd2_helper(self,r,evid):

        result = r.infer(['utility'],evid)
        total = 0
        for atom in result:
            reward = eval(str(atom[0].arguments[0]).strip('\"'))
            total += reward * atom[1]
        return total


    def max_walk_sat_exact_sdd_2(self):
        solution = ""
        utility = -float("inf")
        r = run.run(self.aspContent,self.args)


        for _ in range(1,self.mt):
            mcASPObj = marginal_mcsat.mcSAT(self.aspContent, "", [], self.xorMode,1)
            mcASPObj.runMCASP()
            firstSample = mcASPObj.getSamples()
            decisionAssignments = self.completeTruthAssignment(firstSample[0])
            tempSoln = decisionAssignments
            tempUtility = self.max_walk_sat_sdd2_helper(r, tempSoln)

            for _ in range(1,self.mf):
                vf=""
                deltCostVf = float("inf")
                orgUtility = self.max_walk_sat_sdd2_helper(r,tempSoln)

                if self.p < random.random():
                    vf = tempSoln[random.randint(0, len(tempSoln) - 1)]
                    newDecisionAssignments = self.flipDecisionAtom(tempSoln,vf)
                    flippedUtility =self.max_walk_sat_sdd2_helper(r,newDecisionAssignments)
                    if flippedUtility != False:
                        deltCostVf = orgUtility - flippedUtility
                else:
                    deltaCost = []
                    for decToFlip in tempSoln:
                        newDecisionAssignments = self.flipDecisionAtom(tempSoln, decToFlip)
                        flippedUtility = self.max_walk_sat_sdd2_helper(r,newDecisionAssignments)
                        if flippedUtility != False:
                            deltaCost.append((decToFlip,orgUtility - flippedUtility))
                    if len(deltaCost) != 0:
                        minDeltaCost = min(deltaCost,key=lambda t:t[1])
                    else:
                        continue

                    vf = minDeltaCost[0]
                    deltCostVf = minDeltaCost[1]

                if deltCostVf<0:
                    tempSoln = self.flipDecisionAtom(tempSoln, vf)
                    tempUtility = tempUtility - deltCostVf

            if tempUtility >utility:
                utility = tempUtility
                solution = tempSoln

        return solution,utility


    ###############################
    ## Exact with CNF SDD
    ################################
    def exact_maxSDD(self,save = None,load=None):
        table = list(itertools.product([False, True], repeat=len(self.grounded_dec_atoms)))
        completeDecisionAssignment = []
        for t in table:
            decAssignmentLine = []
            for i in range(0, len(self.grounded_dec_atoms)):
                tuple = (self.grounded_dec_atoms[i], t[i])
                decAssignmentLine.append(tuple)
            completeDecisionAssignment.append(decAssignmentLine)

        completeUitlity = []
        inferObj = sdd_infer.sddInference("", ['utility'])
        inferObj.args = self.args
        if load is None and save is None:
            inferObj.sddConstructorFromLPMLN(self.aspContent)
        elif load is None and save is not None:
            inferObj.sddConstructorFromLPMLN(self.aspContent,True,save)
        else:
            inferObj.sddConstructorFromFile(load)

        if self.args.verbosity >0:
            print("There are ",str(len(completeDecisionAssignment))," different decisions, start to evaluate them one by one.")
        counter = 0
        for decAssiment in completeDecisionAssignment:
            counter+=1
            if self.args.verbosity > 4:
                print("Start evaluating decision: ", counter)
            elif self.args.verbosity > 0:
                if counter == int(0.25*len(completeDecisionAssignment)):
                    print("1/4 is done!")
                elif counter == int(0.5*len(completeDecisionAssignment)):
                    print("1/2 is done!")
                elif counter == int(0.75*len(completeDecisionAssignment)):
                    print("3/4 is done!")
            evidence = self.buildASPEvidenceForm(decAssiment)
            if self.satisfiableChecker(evidence):
                inferObj.queryAtoms = []
                result = inferObj.inference(decAssiment)
                total = 0
                for atom in result:
                    reward = eval(str(atom[0].arguments[0]).strip('\"'))
                    total += reward * atom[1]
                completeUitlity.append((decAssiment,total))
        maxDec = max(completeUitlity,key=lambda item:item[1])
        return maxDec

    ###############################
    ## ExactMAX with lp2SDD
    ################################
    def exact_maxSDD_2(self):
        table = list(itertools.product([False, True], repeat=len(self.grounded_dec_atoms)))
        completeDecisionAssignment = []
        for t in table:
            decAssignmentLine = []
            for i in range(0, len(self.grounded_dec_atoms)):
                tuple = (self.grounded_dec_atoms[i], t[i])
                decAssignmentLine.append(tuple)
            completeDecisionAssignment.append(decAssignmentLine)

        completeUitlity = []
        r = run.run(self.aspContent,self.args)
        r.args = self.args
        if self.args.verbosity >0:
            print("There are ",str(len(completeDecisionAssignment))," different decisions, start to evaluate them one by one.")
        counter = 0
        for decAssiment in completeDecisionAssignment:
            counter+=1
            if self.args.verbosity > 4:
                print("Start evaluating decision: ", counter)
            elif self.args.verbosity > 0:
                if counter == int(0.25*len(completeDecisionAssignment)):
                    print("1/4 is done!")
                elif counter == int(0.5*len(completeDecisionAssignment)):
                    print("1/2 is done!")
                elif counter == int(0.75*len(completeDecisionAssignment)):
                    print("3/4 is done!")


            evidence = self.buildASPEvidenceForm(decAssiment)
            if self.satisfiableChecker(evidence):
                result = r.infer(['utility'],decAssiment)
                total = 0
                for atom in result:
                    reward = eval(str(atom[0].arguments[0]).strip('\"'))
                    total += reward * atom[1]
                completeUitlity.append((decAssiment,total))
        if self.args.verbosity > 0:
            print("All done! Printing final decision.")
        maxDec = max(completeUitlity,key=lambda item:item[1])
        return maxDec


    ###############################
    ## ExactMAX with Exact Probablity Calculation
    ################################
    def exact_max(self):
        table = list(itertools.product([False, True], repeat=len(self.grounded_dec_atoms)))
        completeDecisionAssignment = []
        for t in table:
            decAssignmentLine = []
            for i in range(0,len(self.grounded_dec_atoms)):
                tuple = (self.grounded_dec_atoms[i],t[i])
                decAssignmentLine.append(tuple)
            completeDecisionAssignment.append(decAssignmentLine)

        completeUitlity = []
        if self.args.verbosity >0:
            print("There are ",str(len(completeDecisionAssignment))," different decisions, start to evaluate them one by one.")
        counter=0
        for decAssiment in completeDecisionAssignment:
            counter+=1
            if self.args.verbosity > 4:
                print("Start evaluating decision: ", counter)
            elif self.args.verbosity > 0:
                if counter == int(0.25*len(completeDecisionAssignment)):
                    print("1/4 is done!")
                elif counter == int(0.5*len(completeDecisionAssignment)):
                    print("1/2 is done!")
                elif counter == int(0.75*len(completeDecisionAssignment)):
                    print("3/4 is done!")
            evidence = self.buildASPEvidenceForm(decAssiment)
            utility = self.expectedExactUtility(evidence)
            if utility:
                utilityTuple = (decAssiment,utility)
                completeUitlity.append(utilityTuple)

        maxUtiltiy = -float("inf")
        solution = False
        for eachUtility in completeUitlity:
            if maxUtiltiy < eachUtility[1]:
                maxUtiltiy = eachUtility[1]
                solution = eachUtility[0]
        return solution,maxUtiltiy

    ###############################
    ## ExactMAX with Sampling Based Calculation
    ################################
    def exact_max_app(self):
        table = list(itertools.product([False, True], repeat=len(self.grounded_dec_atoms)))
        completeDecisionAssignment = []
        for t in table:
            decAssignmentLine = []
            for i in range(0,len(self.grounded_dec_atoms)):
                tuple = (self.grounded_dec_atoms[i],t[i])
                decAssignmentLine.append(tuple)
            completeDecisionAssignment.append(decAssignmentLine)

        completeUitlity = []
        if self.args.verbosity >0:
            print("There are ",str(len(completeDecisionAssignment))," different decisions, start to evaluate them one by one.")
        counter=0
        for decAssiment in completeDecisionAssignment:
            counter+=1
            if self.args.verbosity > 4:
                print("Start evaluating decision: ", counter)
            elif self.args.verbosity > 0:
                if counter == int(0.25*len(completeDecisionAssignment)):
                    print("1/4 is done!")
                elif counter == int(0.5*len(completeDecisionAssignment)):
                    print("1/2 is done!")
                elif counter == int(0.75*len(completeDecisionAssignment)):
                    print("3/4 is done!")

            evidence = self.buildASPEvidenceForm(decAssiment)
            utility = self.expectedUtility(evidence)
            if utility:
                utilityTuple = (decAssiment,utility)
                completeUitlity.append(utilityTuple)

        maxUtiltiy = -float("inf")
        solution = False
        for eachUtility in completeUitlity:
            if maxUtiltiy < eachUtility[1]:
                maxUtiltiy = eachUtility[1]
                solution = eachUtility[0]
        return solution,maxUtiltiy




