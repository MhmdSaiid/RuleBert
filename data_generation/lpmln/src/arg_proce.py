
# !/usr/bin/python
import argparse
import sys
import os
from math import exp
from subprocess import Popen, PIPE


def print_error(error, dest):
    if error is not None:
        if dest is not None:
            print_output(error, dest)
        else:
            print (sys.stderr, "*** Error \n")
            print ( sys.stderr, error)




def main():
    parser = argparse.ArgumentParser(description='LPMLN2ASP')
    parser.add_argument('-i', help='input file for inferencing. [REQUIRED]', nargs=1)
    parser.add_argument('-l', help='input file for learning in general algorithm. [REQUIRED]', nargs=1)
    parser.add_argument('-dt', help='input file for decision theory program. [REQUIRED]', nargs=1)
    parser.add_argument('-ls', help='input file for learning with simple lpmln program with in-comlete evidence. [REQUIRED]', nargs=1)
    parser.add_argument('-lsc', help='input file for learning with complete evidence mode. [REQUIRED]', nargs=1)
    parser.add_argument('-e', help='evidence file', nargs=1)
    parser.add_argument('-r', help='output file. If not provided output would be to STDOUT', nargs=1)
    parser.add_argument('-q', help='query predicate.', nargs=1)
    parser.add_argument('-clingo',help='clingo options passed as it is to the solver. Pass all clingo options in \'single quotes\'',nargs=1)
    parser.add_argument('-hr', help='[FALSE] Translate hard rules', action="store_true", default=False)
    parser.add_argument('-all', help='Display probability of all stable models.', action="store_true", default=False)
    parser.add_argument('-mf', help='[1000] Multiplying factor for weak constraints value', nargs=1)
    parser.add_argument('-d', help='[FALSE] Debug. Print all debug information', action="store_true", default=False)
    parser.add_argument('-approximate', help='approximate inference',action="store_true", default=False)
    parser.add_argument('-sdd', help='use SDD',action="store_true", default=False)

    parser.add_argument('-xormode', help='XOR Sampler Mode [Default : 0]', nargs=1)
    parser.add_argument('-ll', help='Learning literations [Default : 50]', nargs=1)
    parser.add_argument('-samp', help='Number of Samples [Default : 50 for leaning, 500 for inferencing]', nargs=1)

    args = parser.parse_args()
    arglist = []





    if args.i is not None and os.path.isfile(args.i[0]):
        arglist.append(args.i[0])
    elif args.l is not None and os.path.isfile(args.l[0]) and args.e is not None and os.path.isfile(args.e[0]):
        arglist.append(args.l[0])
    elif args.ls is not None and os.path.isfile(args.ls[0]) and args.e is not None and os.path.isfile(args.e[0]):
        arglist.append(args.ls[0])
    elif args.lsc is not None and os.path.isfile(args.lsc[0]) and args.e is not None and os.path.isfile(args.e[0]):
        arglist.append(args.lsc[0])
    elif args.dt is not None and os.path.isfile(args.dt[0]):
        arglist.append('--work-type=decision')
    else:
        print_error("Check input file.", None)
        parser.print_help()
        sys.exit(0)


    if args.mf is not None:
        arglist.append('--mf=' + args.mf[0])
    else:
        arglist.append('--mf=1000')
    if args.xormode is not None:
        arglist.append('--xormode='+args.xormode[0])
    if args.l is not None and args.e is not None:
        arglist.append('--work-type=learn')
    if args.ls is not None and args.e is not None:
        arglist.append('--work-type=learns')
    if args.lsc is not None and args.e is not None:
        arglist.append('--work-type=learnsc')

    if args.all is False and args.q is not None:
        arglist.append('-c quiet=true')
        arglist.append('--quiet')

    if args.all is True:
        arglist.append('--work-type=query')

    if args.q is not None:
        arglist.append('-c q=' + args.q[0].replace(',', '__LP__'))
        arglist.append('--work-type=query')

    if args.hr is True:
        arglist.append('--tr-hr=true')

    if args.clingo is not None:
        arglist.append(args.clingo[0].strip("'"))

    if args.e is not None and os.path.isfile(args.e[0]):
        arglist.append('--evidence='+args.e[0])

    if args.approximate is True:
        arglist.append('--work-type=approximate')
    if args.sdd is True:
        arglist.append('--work-type=sdd')
    if args.ll is not None:
        arglist.append('--learning-literation='+args.ll[0])
    if args.samp is not None:
        arglist.append('--mcasp-sample='+args.samp[0])


    return arglist




def print_output(output, dest):
    if dest is not None:
        with open(dest[0], "w") as outputFile:
            outputFile.write(output)
    else:
        print(output)

