# python groundComp.py 2_cnf_atoms.txt 2-3_cnf.txt > 2-4_atom2idx.txt

import sys


class groundComp:
    def __init__(self, content_1, content_2):
        self.mapping = {}
        self.content_1 = content_1
        self.content_2 = content_2

    def striplist(self,l):
        return([x.strip() for x in l])
    def atom2idx_gen(self):
        lines1 = self.content_1
        lines2 = self.content_2

        lines1 = lines1.split("\n")
        lines2 = lines2.split("\n")


        if len(lines1)!=len(lines2):
            print("Error! Line numbers are not equivalent.")
            sys.exit()
        if lines1[0].startswith('p'):
            if lines2[0].startswith('smt'):
                cnf = self.striplist(lines1[1:])
                smt = self.striplist(lines2[1:])
            else:
                print("Error!")
                sys.exit()
        elif lines1[0].startswith('smt'):
            if lines2[0].startswith('p'):
                cnf = self.striplist(lines2[1:])
                smt = self.striplist(lines1[1:])
            else:
                print("Error!")
                sys.exit()
        else:
            print("Error!")
            sys.exit()

        # Generate mapping
        for i in range(len(cnf)):
            cnf_literals = cnf[i].split()
            smt_literals = smt[i].split()
            for j in range(len(cnf_literals)):
                if smt_literals[j][0] == '-':
                    smt_literals[j] = smt_literals[j][1:]
                atom = smt_literals[j]
                if cnf_literals[j] != smt_literals[j] and atom not in self.mapping:
                    self.mapping[atom] = cnf_literals[j].replace("-", "")
        return self.mapping
