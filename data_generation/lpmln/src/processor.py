import os
import time
from _io import StringIO
from src import marginal_mcsat
from src import learn_mcsat
from src import exactProbInfer
from src import sdd_infer
from src import learn_sdd
from src import decision_cal
from src import run
from src import learn_simplelpmln_compEvid
from src import learn_simplelpmln_incompEvid
from src import marginal_coherent
from src import nn_processor
from src.klpmln import MVPP
from src.lpmln_parser import lpmln_parser

def beginSolve(args,type):
    if type=="infer":
        infer = infer_processor(args)
        infer.solve()
    elif type=="decision":
        dec = dec_process(args)
        dec.solve()
    elif type=="learn":
        learn = learn_process(args)
        learn.solve()


class infer_processor(lpmln_parser):
    def __init__(self,args):
        self.arg_list = {
            'file_name': args.Input_File,
            'evidence': [],
            'all': args.all,
            'mf': args.mf[0],
            'query': [],
            'xor_mode': args.xormode[0],
            'sample': args.samp[0],
            'mode':"",
            'hard':args.hard,
            'sddPara':[],
            'nn_indictor':args.nn

        }
        self.args = args
        if args.e is None:
            self.arg_list['evidence'] = []
        else:
            self.arg_list['evidence'] = args.e

        if len(args.q) == 0:
            self.arg_list['query'] = []
        else:
            self.arg_list['query'] =args.q[0].split(',')


        if args.sdd:
            self.arg_list['mode'] = "sdd"
        elif args.sdd2:
            self.arg_list['mode'] = "sdd2"
        elif (args.exact or self.arg_list['query'] or args.hard or args.all) and not args.mcasp and not args.kcor:
            self.arg_list['mode'] = "ex"
        elif args.mcasp:
            self.arg_list['mode'] = "mcasp"
        elif args.kcor:
            self.arg_list['mode'] = "kcor"
        else:
            self.arg_list['mode'] = "map"

        if args.save is None and args.load is None:
            self.arg_list['sddPara'] = [0,0]    #Neither Construct from file or save file
        elif args.save is None:
            self.arg_list['sddPara'] = [1,args.load[0]] # Load from file
        else:
            self.arg_list['sddPara'] = [2,args.save[0]] #Save SDD




    def solve(self):
        wholeInput = ""
        evidenceInput = ""

        if self.arg_list['nn_indictor']:
            wholeInput += nn_processor.process(self.arg_list['file_name'][-1])
            self.arg_list['file_name']  = self.arg_list['file_name'][:-1]

        for lpmln_file in self.arg_list['file_name']:
            with open(lpmln_file, 'r') as lpmln_content:
                wholeInput += lpmln_content.read()
            lpmln_content.close()

        for evid_file in self.arg_list['evidence']:
            with open(evid_file, 'r') as evidence_content:
                evidenceInput += evidence_content.read()
            evidence_content.close()


        if self.arg_list['mode'] == "sdd" :
            content = self.lpmln_to_asp_sdd_parser(wholeInput, False, self.arg_list['mf'])
            parsed_lpmln = self.asp_domain_2_asp_parser(content)
        elif self.arg_list['mode'] == "mcasp":
            content = self.lpmln_to_lpmln_neg_parser(wholeInput,self.arg_list['mf'])
            content = self.lpmln_to_asp_parser(content,False,self.arg_list['mf'])
            parsed_lpmln = self.asp_domain_2_asp_parser(content)
        elif self.arg_list['mode'] == "map":
            content = self.lpmln_to_asp_parser(wholeInput, self.arg_list['hard'], self.arg_list['mf'],True)
            parsed_lpmln = self.asp_domain_2_asp_parser(content)
        else:
            content = self.lpmln_to_asp_parser(wholeInput,self.arg_list['hard'],self.arg_list['mf'],True)
            parsed_lpmln = self.asp_domain_2_asp_parser(content)

        if self.args.verbosity>4:
            print("================== Parsed ASP Program ======================")
            print(parsed_lpmln)

        if self.arg_list['mode'] == "map":
            with open('asp.pl', 'w') as fw:
                fw.write(parsed_lpmln+evidenceInput)
            if self.arg_list['hard']:
                command = "clingo " + os.getcwd() + "/asp.pl --opt-mode=enum 0"
            else:
                command = "clingo " + os.getcwd() + "/asp.pl "
            os.system(command)
        elif self.arg_list['mode'] == "ex":
            warn_option = "--warn=none"
            thread_option = "-t 4"
            enumerateAll = "--opt-mode=enum"
            listAll = "0"
            warn = "--warn=no-atom-undefined"
            options = [warn_option, thread_option, enumerateAll, listAll, warn]
            exactInfer = exactProbInfer.exactProbInfer(parsed_lpmln+evidenceInput, options, self.arg_list['query'], self.arg_list['hard'],self.arg_list['all'])
            if self.arg_list['hard']:
                with open('asp.pl', 'w') as fw:
                    fw.write(parsed_lpmln + evidenceInput)
                command = "clingo " + os.getcwd() + "/asp.pl --opt-mode=enum 0"
                os.system(command)
            exactInfer.solve()
        elif self.arg_list['mode'] == "mcasp":
            mcASPObj = marginal_mcsat.mcSAT(parsed_lpmln, evidenceInput, self.arg_list['query'],self.arg_list['xor_mode'], self.arg_list['sample'])
            mcASPObj.args = self.args
            mcASPObj.runMCASP()
            mcASPObj.printQuery()
        elif self.arg_list['mode'] == "kcor":
            marginal_coherent_obj = marginal_coherent.Marginal_coherent(parsed_lpmln, evidenceInput,self.arg_list['query'],self.arg_list['sample'])
            marginal_coherent_obj.args = self.args
            marginal_coherent_obj.run()
            marginal_coherent_obj.printQuery()

        elif self.arg_list['mode'] == "sdd":
            inferObj = sdd_infer.sddInference(evidenceInput, self.arg_list['query'])
            inferObj.args = self.args
            if self.arg_list['sddPara'][0] == 0:
                inferObj.sddConstructorFromLPMLN(parsed_lpmln, False)
                inferObj.inferencePrtQuery()
            elif self.arg_list['sddPara'][0] == 1:
                inferObj.sddConstructorFromFile(self.arg_list['sddPara'][1])
                inferObj.inferencePrtQuery()
            else:
                inferObj.sddConstructorFromLPMLN(parsed_lpmln,True,self.arg_list['sddPara'][1])
                inferObj.inferencePrtQuery()
        else:
            r = run.run(parsed_lpmln,self.args)
            result = r.infer(self.arg_list['query'],evidenceInput)
            for r in result:
                print(r[0].to_string(),": ", r[1])



class learn_process(lpmln_parser):

    def __init__(self,args):
        self.arg_list = {
            'file_name': args.Input_File,
            'observation': args.o,
            'xor_mode': args.xormode[0],
            'sample': args.samp[0],
            'sdditer':args.sdditer[0],
            'mcsatiter': args.mcsatiter[0],
            'lr':args.lr[0],
            'pretrain':args.pretrain[0],
            'mode':"",
            'sddPara':[],
            'complete':args.complete
        }
        self.args = args

        if args.sdd:
            self.arg_list['mode'] = "sdd"
        elif args.sdd2:
            self.arg_list['mode'] = "sdd2"
        elif args.kcor:
            self.arg_list['mode'] = "kcor"
        elif args.mvpp:
            self.arg_list['mode'] = "mvpp"
        elif args.mcasp:
            self.arg_list['mode'] = "mcasp"


        if args.save is None and args.load is None:
            self.arg_list['sddPara'] = [0,0]    #Neither Construct from file or save file
        elif args.save is None:
            self.arg_list['sddPara'] = [1,args.load[0]] # Load from file
        else:
            self.arg_list['sddPara'] = [2,args.save[0]] #Save SDD

    def solve(self):
        wholeInput = ""
        evidenceInput = ""
        for lpmln_file in self.arg_list['file_name']:
            with open(lpmln_file, 'r') as lpmln_content:
                wholeInput += lpmln_content.read()
            lpmln_content.close()

        for evid_file in self.arg_list['observation']:
            with open(evid_file, 'r') as evidence_content:
                evidenceInput += evidence_content.read()
            evidence_content.close()



        evidence_input_list = [x.strip() for x in evidenceInput.split('#evidence') if x.strip()!='']
        if evidence_input_list is not None:
            evidenceInput = evidence_input_list

        if self.arg_list['mode'] == "sdd":
            if self.arg_list['sddPara'][0] == 0:
                l = learn_sdd.learn_sdd(wholeInput, evidenceInput,None,None,self.arg_list['lr'],self.arg_list['sdditer'])

            elif self.arg_list['sddPara'][0] == 1:  #load from file
                l = learn_sdd.learn_sdd(wholeInput, evidenceInput, None,self.arg_list['sddPara'][1],self.arg_list['lr'],self.arg_list['sdditer'])

            else:# Save
                l = learn_sdd.learn_sdd(wholeInput, evidenceInput,self.arg_list['sddPara'][1],None,self.arg_list['lr'],self.arg_list['sdditer'])
            l.args = self.args
            l.learn()

        elif self.arg_list['mode'] == "sdd2":
            if self.arg_list['sddPara'][0] == 0:
                l = learn_sdd.learn_sdd(wholeInput, evidenceInput,None,None,self.arg_list['lr'],self.arg_list['sdditer'])
            elif self.arg_list['sddPara'][0] == 1:  #load from file
                l = learn_sdd.learn_sdd(wholeInput, evidenceInput, None,self.arg_list['sddPara'][1],self.arg_list['lr'],self.arg_list['sdditer'])
            else:# Save
                l = learn_sdd.learn_sdd(wholeInput, evidenceInput,self.arg_list['sddPara'][1],None,self.arg_list['lr'],self.arg_list['sdditer'])
            l.args = self.args
            l.learn(1)
        elif  self.arg_list['mode'] == "mvpp":
            mvpp_obj = MVPP(wholeInput,self.args)
            mvpp_obj.learn(evidence_input_list, num_of_samples=self.arg_list['sample'], lr=self.arg_list['lr'], thres=0.00001,max_iter=self.arg_list['mcsatiter'],num_pretrain = self.arg_list['pretrain'])

            if self.args.verbosity>=6:
                for i,evid in enumerate(evidence_input_list):
                    if evid!='':
                        print("Prediction on observation ",i," probability is: ", mvpp_obj.inference_obs_exact(obs=evid))
        elif self.arg_list['mode'] == "kcor":
            if self.arg_list['complete']:
                if self.args.verbosity>4:
                    print("Learning simple LPMLN from complete interpretation")
                lsc = learn_simplelpmln_compEvid.learn_simple_comp_evid(wholeInput,evidenceInput,self.arg_list['mcsatiter'],self.arg_list['lr'])
                lsc.args = self.args
                lsc.learn()
            else:
                if self.args.verbosity>4:
                    print("Learning simple LPMLN from partial interpretation")
                ls = learn_simplelpmln_incompEvid.learn_simple_incomp_evid(wholeInput,evidenceInput,self.arg_list['xor_mode'],self.arg_list['mcsatiter'],self.arg_list['sample'],self.arg_list['lr'])
                ls.args = self.args
                ls.learn()

        else:
            if self.arg_list['complete']:
                learnObj = learn_mcsat.learn_general_ga_mcasp(wholeInput, evidenceInput,self.arg_list['xor_mode'],self.arg_list['mcsatiter'],self.arg_list['sample'],self.arg_list['lr'],True)
            else:
                learnObj = learn_mcsat.learn_general_ga_mcasp(wholeInput, evidenceInput,self.arg_list['xor_mode'],self.arg_list['mcsatiter'],self.arg_list['sample'],self.arg_list['lr'])

            learnObj.args = self.args
            learnObj.learn()


class dec_process(lpmln_parser):

    def __init__(self,args):
        self.arg_list = {
            'file_name': args.Input_File,
            'query': args.q,
            'xor_mode': args.xormode[0],
            'sample': args.samp[0],
            'mode':"",
            'alg': "",
            'sddPara':[],
            'dec_predicate':args.dec[0].split(','),
        }
        self.args = args

        if args.maxwalksat:
            self.arg_list['alg'] = "maxwalksat"
        else:
            self.arg_list['alg'] = "exmax"


        if args.sdd:
            self.arg_list['mode'] = "sdd"
        elif args.sdd2:
            self.arg_list['mode'] = 'sdd2'
        elif args.mcasp:
            self.arg_list['mode'] = "mcasp"
        elif args.ex:
            self.arg_list['mode'] = "ex"


        if args.save is None and args.load is None:
            self.arg_list['sddPara'] = [0,0]    #Neither Construct from file or save file
        elif args.save is None:
            self.arg_list['sddPara'] = [1,args.load[0]] # Load from file
        else:
            self.arg_list['sddPara'] = [2,args.save[0]] #Save SDD


    def solve(self):
        wholeInput = ""
        queryInput = ""
        for lpmln_file in self.arg_list['file_name']:
            with open(lpmln_file, 'r') as lpmln_content:
                wholeInput += lpmln_content.read()
            lpmln_content.close()

        if self.arg_list['query'] is not None:
            for q_file in self.arg_list['query']:
                with open(q_file, 'r') as q_content:
                    queryInput += q_content.read()
                    q_content.close()

        if self.arg_list['mode'] == "sdd" or self.arg_list['mode'] =="sdd2":
            content = self.lpmln_to_asp_sdd_parser(wholeInput)
            parsed_lpmln = self.asp_domain_2_asp_parser(content)

        elif self.arg_list['mode'] == "mcasp":
            content = self.lpmln_to_lpmln_neg_parser(wholeInput)
            content = self.lpmln_to_asp_parser(content)
            parsed_lpmln = self.asp_domain_2_asp_parser(content)
        else:
            content = self.lpmln_to_asp_parser(wholeInput)
            parsed_lpmln = self.asp_domain_2_asp_parser(content)


        if self.args.verbosity>4:
            print("================== Parsed ASP Program Start ======================")
            print(parsed_lpmln)
            print("================== Parsed ASP Program End ======================")

        if self.arg_list['mode'] == "ex":
            print("mode=ex")
            dt = decision_cal.decision_cal(parsed_lpmln,self.arg_list['dec_predicate'])
            dt.args = self.args
            if queryInput !="":
                result = dt.expectedExactUtility(queryInput)
            else:
                if self.arg_list['alg'] == "maxwalksat":
                    result = dt.max_walk_sat_exact()
                else:
                    result = dt.exact_max()
        ##############################################################3
        elif self.arg_list['mode'] == "mcasp":
            print("mode=mcasp")
            dt = decision_cal.decision_cal(parsed_lpmln,self.arg_list['dec_predicate'] , self.arg_list['xor_mode'],self.arg_list['sample'])
            dt.args = self.args

            if queryInput !="":
                result = dt.expectedUtility(queryInput)
            else:
                if self.arg_list['alg'] == "maxwalksat":
                    result = dt.max_walk_sat()
                else:
                    result = dt.exact_max_app()
        ##############################################################3
        elif self.arg_list['mode'] == "sdd":
            print("mode=sdd")
            dt = decision_cal.decision_cal(parsed_lpmln, self.arg_list['dec_predicate'])
            dt.args = self.args

            if self.arg_list['sddPara'][0] == 0:
                if queryInput !="":
                    result = dt.expectedExactUtility_SDD(queryInput)
                else:
                    if self.arg_list['alg'] == "maxwalksat":
                        result = dt.max_walk_sat_exact_sdd()
                    else:
                        result = dt.exact_maxSDD()

            elif self.arg_list['sddPara'][0] == 1:  # load from file
                if queryInput !="":
                    result = dt.expectedExactUtility_SDD(queryInput,None,self.arg_list['sddPara'][1])
                else:
                    if self.arg_list['alg'] == "maxwalksat":
                        result = dt.max_walk_sat_exact_sdd(None,self.arg_list['sddPara'][1])
                    else:
                        result = dt.exact_maxSDD(None,self.arg_list['sddPara'][1])
            else:
                if queryInput !="":
                    result = dt.expectedExactUtility_SDD(queryInput, self.arg_list['sddPara'][1],None)
                else:
                    if self.arg_list['alg'] == "maxwalksat":
                        result = dt.max_walk_sat_exact_sdd(self.arg_list['sddPara'][1], None)
                    else:
                        result = dt.exact_maxSDD(self.arg_list['sddPara'][1], None)
        ##############################################################3
        elif self.arg_list['mode'] == "sdd2":
            print("mode=sdd2")
            dt = decision_cal.decision_cal(parsed_lpmln, self.arg_list['dec_predicate'])
            dt.args = self.args

            if queryInput!="":
                result = dt.expectedExactUtility_SDD_2(queryInput)
            else:
                result = dt.max_walk_sat_exact_sdd_2()
        else:

            '''if self.arg_list['alg'] == "maxwalksat":
                dt = decision_cal.decision_cal(parsed_lpmln, self.arg_list['dec_predicate'])
                dt.args = self.args
                dt.max_walk_sat_exact_sdd_2()
            else:'''
            dt = decision_cal.decision_cal(parsed_lpmln, self.arg_list['dec_predicate'])
            dt.args = self.args

            if self.args.verbosity>4:
                print("Finished building SDD")

            result = dt.exact_maxSDD_2()
            if self.args.verbosity>4:
                print("Start finding optimial actions")
        print(result)










