
import subprocess
from src import groundComp
import time
import math
from pysdd.sdd import SddManager, Vtree, WmcManager
import clingo
import random
import datetime
import json
import os.path
import sys
import math
import sys

class sddInference(object):
    def __init__(self,observation,queryPredicates):
        self.content = ""
        self.sddfName = ""
        self.bytesContent = ""
        self.observation = observation
        self.queryPredicates = queryPredicates
        self.nameMappingDic = {}
        self.sdd = ""
        self.formula = ""
        self.queryAtoms = []
        self.saveSDD = ""
        self.evidenceAssignment = self.parseEvidence()
        self.modelDics = {}
        self.args = []


    def sddConstructorFromLPMLN(self,content,saveSDD = False, savedSDDName=None):
        self.content = content
        self.bytesContent = str.encode(self.content)
        self.saveSDD = saveSDD
        self.saveSDDName = savedSDDName
        fn_gringo = os.path.join(os.path.dirname(__file__), '../binSupport/gringo')
        p = subprocess.Popen([fn_gringo], shell=True, stdout=subprocess.PIPE, stdin=subprocess.PIPE)
        p.stdin.write(self.bytesContent)
        gringo_out = p.communicate()[0]
        p.stdin.close()
        p.stdout.close()
        if self.args.verbosity > 4:
            print("Grounding Done! ")
        fn_cmodels = os.path.join(os.path.dirname(__file__), '../binSupport/cmodels')
        p = subprocess.Popen([fn_cmodels+' -cdimacs'], shell=True, stdout=subprocess.PIPE, stdin=subprocess.PIPE)
        p.stdin.write(gringo_out)
        cmodels_out_Name = p.communicate()[0]
        p.stdin.close()
        p.stdout.close()
        #print("Completion_1 Done! ")



        p = subprocess.Popen([fn_cmodels+' -dimacs'], shell=True, stdout=subprocess.PIPE, stdin=subprocess.PIPE)
        p.stdin.write(gringo_out)
        cmodels_out_no_name = p.communicate()[0]
        p.stdin.close()
        p.stdout.close()

        if self.args.verbosity > 4:
            print("Completion Done! ")

        gp = groundComp.groundComp(cmodels_out_Name.decode(), cmodels_out_no_name.decode())
        self.nameMappingDic = gp.atom2idx_gen()
        if self.args.verbosity > 4:
            print("Name Mapping Done!")

        start = time.time()
        self.sdd, self.formula = SddManager.from_cnf_string(cmodels_out_no_name.decode())

        if self.args.verbosity > 4:
            print("SDD Done! Time For Construct SDD: ", str(time.time() - start))

        if self.saveSDD:
            return self.save()


    def fillQueryAtoms(self):
        for q in self.queryPredicates:
            for key, value in self.nameMappingDic.items():
                symbol = clingo.parse_term(key)
                if (q == symbol.name):
                    self.queryAtoms.append([symbol,0.0])


    def sddConstructorFromFile(self,sddFName):
        self.sddfName = sddFName
        if os.path.isfile(self.sddfName+".sdd") and os.path.isfile(self.sddfName+".vtree") and os.path.isfile(self.sddfName+".map") :
            with open(self.sddfName + ".map", "r") as f:
                self.nameMappingDic = json.load(f)
            vtree = Vtree.from_file((self.sddfName+".vtree").encode())
            self.sdd = SddManager.from_vtree(vtree)
            self.formula = self.sdd.read_sdd_file((self.sddfName+".sdd").encode())
        else:
            print("Please provide correct files")
            sys.exit()

    def iniWMC(self):
        wmc = self.formula.wmc(log_mode=True)
        for key, items in self.nameMappingDic.items():
            symbol = clingo.parse_term(key)
            if (symbol.name == "unsat"):
                wmc.set_literal_weight(-eval(str(items)), eval(str(symbol.arguments[1]).strip('\"')))
        orgWMC = math.exp(wmc.propagate())

        return wmc,orgWMC


    def ini_WMC_with_observation(self):
        wmc = self.formula.wmc(log_mode=True)
        for key, items in self.nameMappingDic.items():
            symbol = clingo.parse_term(key)
            if (symbol.name == "unsat"):
                wmc.set_literal_weight(-eval(str(items)), eval(str(symbol.arguments[1]).strip('\"')))
            for tuple in self.evidenceAssignment:
                if str(symbol) == str(tuple[0]):
                    if tuple[1] == 0:
                        wmc.set_literal_weight(eval(str(items)),-float("inf"))

                    else:
                        wmc.set_literal_weight(-eval(str(items)),-float("inf"))

        orgWMC = math.exp(wmc.propagate())

        return wmc,orgWMC


    def learn_iniWMC(self):
        wmc = self.formula.wmc(log_mode=False)
        for key, items in self.nameMappingDic.items():
            symbol = clingo.parse_term(key)
            if (symbol.name == "unsat"):
                wmc.set_literal_weight(-eval(str(items)), math.exp(eval(str(symbol.arguments[1]).strip('\"'))))
        orgWMC = wmc.propagate()

        return wmc,orgWMC


    def learn_ini_WMC_with_observation(self, parsed_observation):
        wmc = self.formula.wmc(log_mode=False)
        for key, items in self.nameMappingDic.items():
            symbol = clingo.parse_term(key)
            if (symbol.name == "unsat"):
                wmc.set_literal_weight(-eval(str(items)), math.exp(eval(str(symbol.arguments[1]).strip('\"'))))
            for tuple in parsed_observation:
                if str(symbol) == str(tuple[0]):
                    if tuple[1] == 0:
                        wmc.set_literal_weight(eval(str(items)),0)
                    else:
                        wmc.set_literal_weight(-eval(str(items)),0)
        orgWMC = wmc.propagate()
        return wmc,orgWMC


    def adjustSDDWeight(self,wmc,adj):
        for key, items in self.nameMappingDic.items():
            symbol = clingo.parse_term(key)
            for tuple in adj:
                if str(symbol) == str(tuple[0]):
                    if tuple[1]:
                        wmc.set_literal_weight(-eval(str(items)), -float("inf"))
                    else:
                        wmc.set_literal_weight(eval(str(items)), -float("inf"))
        orgWMC = math.exp(wmc.propagate())
        return wmc, orgWMC



    def parseEvidence(self,obs = None):
        if obs == None:
            evidSet = self.observation.split('.')
        else:
            evidSet = obs.split('.')
        atomAssignment = []
        for e in evidSet:
            e = e.strip()
            if e == "":
                continue
            elif "not" in e:
                atomAssignment.append((clingo.parse_term((e.split("not")[1].strip(" "))),1))
            else:
                atomAssignment.append((clingo.parse_term((e.split(":-")[1].strip(" "))),0))
        return atomAssignment



    def inference(self,adjustWeight=None):
        self.fillQueryAtoms()
        for queryAtom in self.queryAtoms:
            if self.observation == "":
                wmc,orgWNC = self.iniWMC()
            else:
                wmc,orgWNC = self.ini_WMC_with_observation()

            if adjustWeight != None:
                wmc, orgWNC = self.adjustSDDWeight(wmc,adjustWeight)
            for key, items in self.nameMappingDic.items():
                symbol = clingo.parse_term(key)
                if (str(symbol) == str(queryAtom[0])):
                    wmc.set_literal_weight(-eval(str(items)), -float("inf"))
                    after = math.exp(wmc.propagate())


                    queryAtom[1] = after/orgWNC
                    break
        return self.queryAtoms
    def inferencePrtQuery(self):
        self.fillQueryAtoms()
        for queryAtom in self.queryAtoms:
            if self.observation == "":
                wmc,orgWNC = self.iniWMC()
            else:
                wmc,orgWNC = self.ini_WMC_with_observation()

            for key, items in self.nameMappingDic.items():
                symbol = clingo.parse_term(key)
                if (str(symbol) == str(queryAtom[0])):
                    wmc.set_literal_weight(-eval(str(items)), -float("inf"))
                    after = math.exp(wmc.propagate())
                    queryAtom[1] = after / orgWNC
                    break
        for atom in self.queryAtoms:
            print(atom)


    def interpretationInfer(self,interpretation):
        wmc, orgWNC = self.iniWMC()
        for atoms in interpretation:
            for key, items in self.nameMappingDic.items():
                symbol = clingo.parse_term(key)
                if (str(symbol) == str(atoms[0])):
                    if atoms[1]:
                        wmc.set_literal_weight(-eval(str(items)), -float("inf"))
                    else:
                        wmc.set_literal_weight(eval(str(items)), -float("inf"))
        after =math.exp(wmc.propagate())
        return after/orgWNC



    def save(self):
        if self.saveSDDName is None:
            name = str(str(datetime.datetime.now()).split(' ')[1])+str(random.random())
        else:
            name = self.saveSDDName
        self.formula.save(str.encode(name+".sdd"))
        self.formula.vtree().save(str.encode(name+".vtree"))
        open(name+".map", "w").close()
        with open(name+".map", "w") as f:
            json.dump(self.nameMappingDic, f)
        print("SDD saved.")
        return name
    def modelCount(self):
        return self.formula.model_count()


    def modelFactoryAllTruth(self):
        for model in self.formula.models():
            eachModel = []
            for key, items in self.nameMappingDic.items():
                if model[eval(str(items))] == 1:
                    eachModel.append((key,True))
                else:
                    eachModel.append((key,False))
            yield eachModel

    def modelFactory(self):
        #self.sdd, self.formula = SddManager.from_cnf_string("CNF Input".decode())
        for model in self.formula.models():
            eachModel = []
            for key, items in self.nameMappingDic.items():
                if model[eval(str(items))] == 1:
                    eachModel.append(key)
            yield eachModel


    def getAllTruthAssignments(self):
        for m in self.modelFactoryAllTruth():
            yield m


    def getModels(self):
        for m in self.modelFactory():
            yield m
    def get_num_models(self):
        self.iniWMC()
