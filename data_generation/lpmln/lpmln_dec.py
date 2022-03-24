#! /home/barbacou/miniconda3/bin/python
import argparse
import os
import sys
import time

from src import processor

class lpmln_dec(object):
    def __init__(self):
        self.decParser = argparse.ArgumentParser(description='LPMLN-Decision Theory', prog='LPMLN2ASP')

    def error_printer(self,message):
        self.decParser.print_help()
        self.decParser.error(message)

    def errorChecker(self, args):
        if len(args.dec) == 0:
            self.error_printer("Please provide decision predicates")
        elif len(args.Input_File)==0:
            self.error_printer("Please provide input file")
        elif not all(os.path.exists(file) for file in args.Input_File):
            self.error_printer("Please provide correct input file path")
        elif args.q is not None and not os.path.exists(args.q[0]):
                self.error_printer("Please provide correct query file path")
        elif args.load is not None:
            if not (os.path.exists(args.load[0]+'.vtree') and os.path.exists(args.load[0]+'.sdd') and os.path.exists(args.load[0]+'.map')):
                self.error_printer("Please provide correct SDD file path")

    def parser(self,argv):
        self.decParser.add_argument('Input_File', help='Input LPMLN file. [REQUIRED]',nargs="*")
        self.decParser.add_argument('-q', help='File for querying expected utility',nargs=1)

        group = self.decParser.add_mutually_exclusive_group()
        group.add_argument('-sdd',help='Inference Method: SDD-CNF Based Exact Probablity Inference.',action="store_true")
        group.add_argument('-ex',help='Inference Method: Exact Probablity Inference. ',action="store_true",default=True)
        group.add_argument('-mcasp',help='Inference Method: Approximate Probablity Inference.[Default]',action="store_true")
        group.add_argument('-sdd2',help='Inference Method: SDD-lp2sdd Based Exact Probablity Inference.',action="store_true", default=False)


        self.decParser.add_argument('-dec',help='Decision Predicates. [Seperated by \',\', required]',nargs=1,default=[],required=True)
        self.decParser.add_argument('-xormode', help='XOR Sampler Mode [Default : 2]', nargs=1, choices=range(0,3),type=int,default=[2])
        self.decParser.add_argument('-samp', help='Number of Samples [Default: 50]', nargs=1, type=int,default=[50])

        group_find_dec = self.decParser.add_mutually_exclusive_group()

        group_find_dec.add_argument('-exmax', help='Finding Decision by The Exact Max Algorithm.[Default]',action="store_true",default=True)
        group_find_dec.add_argument('-maxwalksat', help='Finding Decision by The Max Walk Sat Algorithm.[Default]',action="store_true",default=False)


        sddInfoGroup = self.decParser.add_mutually_exclusive_group()
        sddInfoGroup.add_argument('-save', help='Saved SDD File Name',nargs=1,type=str)
        sddInfoGroup.add_argument('-load', help='Load SDD From a File',nargs=1,type=str)
        self.decParser.add_argument('-verbosity', '-v', default=0, type=int, help='set verbosity')
        self.decParser.add_argument('-timeout', '-t', default=None, type=float, help='set timeout for compilation')


        self.args = self.decParser.parse_args(argv)


        self.errorChecker(self.args)
        processor.beginSolve(self.args, "decision")


if __name__ == "__main__":
    start = time.time()

    dec = lpmln_dec()
    dec.parser(sys.argv[1:])

    if dec.args.verbosity >0:
        print("Total time: ",time.time()-start)





#print(decParser.print_help())