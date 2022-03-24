def get_parser():
    import argparse
    from argparse import RawTextHelpFormatter

    parser = argparse.ArgumentParser(description='lp2sdd: Compilation of logic programs',
                                     formatter_class=RawTextHelpFormatter)
    parser.add_argument('--inputFiles', '-i', default=None, help='provide input as file '
                        '(if none is given, stdin is used)', nargs='+', metavar="FILE")
    parser.add_argument('--timeout', '-t', default=None, type=float, help='set timeout for compilation')
    parser.add_argument('--verbosity', '-v', default=0, type=int, help='set verbosity')
    parser.add_argument('--version', action='version', version='%(prog)s 1.0')
    parser.add_argument('--semantics', '-s', default="stable", help="The semantics to be used",
                        choices=["stable", "well-founded", "completion"])
    parser.add_argument('--language', '-l', default="lparse", help="The language in which input is specified used",
                        choices=["lparse", "problog"])
    parser.add_argument('--install', '-I', action='store_true', help='Install the sdd library')
    parser.add_argument('--inference', '-d', default="count_models",
                        help="The inference to perform on the program(s):"
                        "\n\t* count_models prints the model count of one input program"
                        "\n\t* count_voc_models prints the model count of one input program projected onto a given "
                        "vocabulary. The vocabulary consists of all symbols occuring in the symbol table"
                        # "\n\t* check_equivalence checks whether two input programs are equivalent. They are required "
                        # "to be programs over the same vocabulary"
                        "\n\t* check_voc_equivalence checks whether the two input programs are equivalent modulo "
                        "variables not occuring in the symbol tables",
                        choices=["count_models",
                                 "count_voc_models",
                                 # "check_equivalence",
                                 "check_voc_equivalence"])
    parser.add_argument('--save-state', '-S', type=str, default=None,
                        help='Only compile the program, perform no inferece. Save state to perform inference later. '
                             'Takes as argument a directory to store the state in.',
                        metavar="DIR", dest="savedir")
    parser.add_argument('--load-state', '-L', type=str,  default=None,
                        help='Load a previously saved state and performs inference on that logic program. Takes as '
                             'argument the directory in which the state is saved',
                        metavar="DIR", dest="loaddir")
    return parser
