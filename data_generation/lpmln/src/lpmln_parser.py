from src import lpmln2lpmlnneg_lexer
from src import lpmln2asp_lexer
from src import lpmln2asp_lexer_sdd
from src import asp_domain_2_asp_lexer

class lpmln_parser(object):

    def __init__(self,debug=False):
        self.debug=debug


    def lpmln_to_lpmln_neg_parser(self,content,mfactor=1000):
        parser = lpmln2lpmlnneg_lexer.lpmln2lpmlnnet(content,mfactor)
        output = parser.parseToNeg()

        #print("First parser. Done!")
        return output



    def lpmln_to_asp_parser(self,content,tranlate_hard_rule=False,mfactor=1000,map=False):

        parser = lpmln2asp_lexer.lpmln2asp(content, tranlate_hard_rule, mfactor,map)
        output = parser.parseToASP()
        #print("Second parser. Done!")
        return output


    def lpmln_to_asp_sdd_parser(self,content,tranlate_hard_rule=False,mfactor=1000):
        parser = lpmln2asp_lexer_sdd.lpmln2asp(content, tranlate_hard_rule, mfactor)
        output = parser.parseToASP()
        #print("Second parser. Done!")
        return output

    def asp_domain_2_asp_parser(self,content):
        parser = asp_domain_2_asp_lexer.domianRemover(content)
        output = parser.parseToRemoveDomain()
        #print("Third parser. Done!")
        return output