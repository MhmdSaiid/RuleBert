#! /home/barbacou/miniconda3/bin/python
import argparse
import os
import sys
import time

from src import processor

class lpmln_learn(object):
    def __init__(self):
        self.learn_arg_parser = argparse.ArgumentParser(description='LPMLN-Learning', prog='LPMLN2ASP')

    def error_printer(self,message):
        self.learn_arg_parser.print_help()
        self.learn_arg_parser.error(message)

    def errorChecker(self, args):

        if len(args.Input_File)==0:
            self.error_printer("Please provide input file")
        elif not all(os.path.exists(file) for file in args.Input_File):
            self.error_printer("Please provide correct input file path")
        elif args.o is not None and not os.path.exists(args.o[0]):
                self.error_printer("Please provide correct observation file path")
        elif (not (args.load is None) or not(args.save is None) and (args.sdd==False)):
            self.error_printer("-load or -save can only use with -sdd option")
        elif args.load is not None:
            if not (os.path.exists(args.load[0]+'.vtree') and os.path.exists(args.load[0]+'.sdd') and os.path.exists(args.load[0]+'.map')):
                self.error_printer("Please provide correct SDD file path")


    def parser(self,argv):
        self.learn_arg_parser.add_argument('Input_File', help='Input LPMLN File. [REQUIRED]',nargs="*")
        self.learn_arg_parser.add_argument('-o', help='Observation File.',nargs=1,required=True)
        group = self.learn_arg_parser.add_mutually_exclusive_group()
        group.add_argument('-sdd',help='Learning Method: SDD-CNF Based Exact Probablity Learning.', default=False, action="store_true")
        group.add_argument('-sdd2',help='Learning Method: SDD-lp2sdd Based Exact Probablity Learning.', default=False, action="store_true")

        group.add_argument('-mcasp',help='Learning Method: Approximate Probablity Learning.',default=True,action="store_true")
        group.add_argument('-kcor',help='Learning Method: Simple LPMLN Learning with Complete Evidence.',default=False,action="store_true")
        group.add_argument('-mvpp', help='Learning Method: MVPP Program Learning.',default=False,action='store_true')

        self.learn_arg_parser.add_argument('-complete', help='Learning from complete interpretaion', default=False, action="store_true")
        self.learn_arg_parser.add_argument('-xormode', help='XOR Sampler Mode [Default : 2]', nargs=1, choices=range(0,3),type=int,default=[2])
        self.learn_arg_parser.add_argument('-samp', help='Number of Samples [Default: 50]', nargs=1, type=int,default=[50])
        self.learn_arg_parser.add_argument('-sdditer', help='Number of Learning Iteration [Default: 1000]', nargs=1, type=int,default=[1000])
        self.learn_arg_parser.add_argument('-mcsatiter', help='Number of Learning Iteration [Default: 50]', nargs=1, type=int,default=[50])
        self.learn_arg_parser.add_argument('-pretrain', help='Number of Pre-train on MVPP Propgram [Default: 5]', nargs=1, type=int,default=[5])

        self.learn_arg_parser.add_argument('-lr', help='Learning Rate [Default: 0.01]', nargs=1, type=float,default=[0.01])

        sddInfoGroup = self.learn_arg_parser.add_mutually_exclusive_group()
        sddInfoGroup.add_argument('-save', help='Saved SDD File Name',nargs=1,type=str)
        sddInfoGroup.add_argument('-load', help='Load SDD From a File',nargs=1,type=str)

        self.learn_arg_parser.add_argument('-verbosity', '-v', default=0, type=int, help='set verbosity')
        self.learn_arg_parser.add_argument('-timeout', '-t', default=None, type=float, help='set timeout for compilation')

        self.args = self.learn_arg_parser.parse_args(argv)

        if self.args.verbosity>4:
            print(self.args)



        self.errorChecker(self.args )
        processor.beginSolve(self.args , "learn")


if __name__ == "__main__":
    start = time.time()

    ll = lpmln_learn()
    ll.parser(sys.argv[1:])

    if ll.args.verbosity >0:
        print("Total time: ",time.time()-start)




