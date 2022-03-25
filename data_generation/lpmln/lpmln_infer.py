#! /home/barbacou/miniconda3/envs/rulebert/bin/python
#! /home/barbacou/miniconda3/envs/rulebert/bin/python
#! /home/barbacou/miniconda3/envs/env_lpmln/bin/python
#! /home/barbacou/miniconda3/envs/env_lpmln/bin/python
#! /home/barbacou/miniconda3/envs/env_lpmln/bin/python
#! /home/barbacou/miniconda3/envs/env_lpmln/bin/python
#! /home/barbacou/miniconda3/envs/env_lpmln/bin/python
#! /home/barbacou/miniconda3/bin/python
import argparse
import os
import sys
import time

from src import processor

class lpmln_infer(object):
    def __init__(self):
        self.inferParser = argparse.ArgumentParser(description='LPMLN-Inference', prog='LPMLN2ASP')

    def error_printer(self,message):
        self.inferParser.print_help()
        self.inferParser.error(message)

    def errorChecker(self, args):
        if (args.sdd or args.mcasp) and len(args.q)==0:
            self.error_printer("No query predicate is provided")
        elif len(args.Input_File)==0:
            self.error_printer("Please provide input file")
        elif not all(os.path.exists(file) for file in args.Input_File):
            self.error_printer("Please provide correct input file path")
        elif args.e is not None:
            if not os.path.exists(args.e[0]):
                self.error_printer("Please provide correct evidence file path")
        elif (args.all and args.mcasp):
            self.error_printer("Approximate probablity inference cannot display probability for all stable models")
        elif (args.hard and args.mcasp) or (args.hard and args.sdd):
            self.error_printer("Translating hard rule can only work with -exact, -map command")
        elif args.load is not None:
            if not (os.path.exists(args.load[0]+'.vtree') and os.path.exists(args.load[0]+'.sdd') and os.path.exists(args.load[0]+'.map')):
                self.error_printer("Please provide correct SDD file path")


    def parser(self,argv):
        self.inferParser.add_argument('Input_File', help='Input LPMLN file. [REQUIRED]',nargs="*")
        self.inferParser.add_argument('-e', help='Evidence file.',nargs=1)

        self.inferParser.add_argument('-nn',help='NN input indicator',action="store_true",default=False)

        group = self.inferParser.add_mutually_exclusive_group()
        group.add_argument('-map',help='Inference Method: MAP Inference. [Default]',action="store_true",default=True)
        group.add_argument('-sdd',help='Inference Method: SDD-CNF Based Exact Probablity Inference.',action="store_true", default=False)
        group.add_argument('-sdd2',help='Inference Method: SDD-lp2sdd Based Exact Probablity Inference.',action="store_true", default=False)

        group.add_argument('-exact',help='Inference Method: Exact Probablity Inference. ',action="store_true",default=False)
        group.add_argument('-mcasp',help='Inference Method: Approximate Probablity Inference.',action="store_true",default=False)
        group.add_argument('-kcor',help='Inference Method: Simple LPMLN Probablity Inference.',action="store_true",default=False)
        group.add_argument('-mvpp',help='Inference Method: MVPP Program Inference.',action="store_true",default=False)


        self.inferParser.add_argument('-q',help='Query Predicates. [Seperated by \',\']',nargs=1,default=[])
        self.inferParser.add_argument('-hard', help='Whether Translate Hard Rules [Work only with -exact, -map command. Default: False]', action="store_true", default=False)
        self.inferParser.add_argument('-all', help='Display Probability of All Stable Models. [Only work with -exact. Default: False]', action="store_true", default=False)
        self.inferParser.add_argument('-xormode', help='XOR Sampler Mode [Default : 2]', nargs=1, choices=range(0,3),type=int,default=[2])
        self.inferParser.add_argument('-samp', help='Number of Samples [Default: 500]', nargs=1, type=int,default=[500])
        self.inferParser.add_argument('-mf', help='Multiplying Factor for Weak Constraints Value[Default: 1000]', nargs=1,default=[1000],type=int)

        sddInfoGroup = self.inferParser.add_mutually_exclusive_group()
        sddInfoGroup.add_argument('-save', help='Saved SDD File Name',nargs=1,type=str)
        sddInfoGroup.add_argument('-load', help='Load SDD From a File',nargs=1,type=str)
        self.inferParser.add_argument('-verbosity', '-v', default=0, type=int, help='set verbosity')
        self.inferParser.add_argument('-timeout', '-t', default=None, type=float, help='set timeout for compilation')

        self.args = self.inferParser.parse_args(argv)

        self.errorChecker(self.args)
        processor.beginSolve(self.args, "infer")


if __name__ == "__main__":


    start = time.time()
    infer = lpmln_infer()
    infer.parser(sys.argv[1:])

    if infer.args.verbosity >0:
        print("Total time: ",time.time()-start)






#print(inferParser.print_help())