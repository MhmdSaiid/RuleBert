from math import exp, expm1, log
import sys
import ply.yacc as yacc
import ply.lex as domain_lex


class domianRemover(object):

    reserved = {
        'not': 'NOT',
    }

    tokens = [
        'DIGIT', 'IDENTIFIER', 'VARIABLE', 'ANONYMOUS', 'STRING',
        'ADD', 'AND', 'EQ', 'AT', 'COLON', 'COMMA', 'CONST', 'COUNT', 'CSP', 'CSP_ADD',
        'CSP_SUB', 'CSP_MUL', 'CSP_LEQ', 'CSP_LT', 'CSP_GT', 'CSP_GEQ', 'CSP_EQ', 'CSP_NEQ', 'DISJOINT', 'DOT',
        'DOTS', 'EXTERNAL', 'FALSE', 'GEQ', 'GT', 'IF', 'INCLUDE', 'INFIMUM', 'LBRACE', 'LBRACK', 'LEQ', 'LPAREN',
        'LT', 'MAX', 'MAXIMIZE', 'MIN', 'MINIMIZE', 'MOD', 'MUL', 'NEQ', 'POW', 'QUESTION', 'RBRACE', 'RBRACK',
        'RPAREN', 'SEM', 'SHOW',
        'EDGE', 'PROJECT', 'HEURISTIC', 'SHOWSIG', 'SLASH', 'SUB', 'SUM', 'SUMP', 'SUPREMUM', 'TRUE', 'BLOCK',
        'VBAR', 'WIF', 'XOR', 'COMMENT', 'WS', 'DOMAIN',
    ]
    tokens += reserved.values()

    t_ANONYMOUS = r'\_'
    t_STRING = r'\"([^\\"\n]|"\\\""|"\\\\"|"\\n")*\"'
    t_WS = r'\s'
    t_ADD = r'\+'
    t_AND = r'\&'
    t_EQ = r'\='
    t_AT = r'\@'
    t_COLON = r'\:'
    t_COMMA = r'\,'
    t_CONST = r'\#const'
    t_COUNT = r'\#count'
    t_CSP = r'\$'
    t_CSP_ADD = r'\$\+'
    t_CSP_SUB = r'\$\-'
    t_CSP_MUL = r'\$\*'
    t_CSP_LEQ = r'\$\<\='
    t_CSP_LT = r'\$\<'
    t_CSP_GT = r'\$\>'
    t_CSP_GEQ = r'\$\>\='
    t_CSP_EQ = r'\$\='
    t_CSP_NEQ = r'\$\!\='
    t_DISJOINT = r'\#disjoint'
    t_DOT = r'\.'
    t_DOTS = r'\.\.'
    t_EXTERNAL = r'\#external'
    t_FALSE = r'\#false'
    t_GEQ = r'\>='
    t_GT = r'\>'
    t_IF = r'\:\-'
    t_INCLUDE = r'\#include'
    t_INFIMUM = r'\#inf'
    t_LBRACE = r'\{'
    t_LBRACK = r'\['
    t_LEQ = r'\<\='
    t_LPAREN = r'\('
    t_LT = r'\<'
    t_MAX = r'\#max'
    t_MAXIMIZE = r'\#maximize'
    t_MIN = r'\#min'
    t_MINIMIZE = r'\#minimize'
    t_MOD = r'\\'
    t_MUL = r'\*'
    t_NEQ = r'\!\='
    t_POW = r'\*\*'
    t_QUESTION = r'\?'
    t_RBRACE = r'\}'
    t_RBRACK = r'\]'
    t_RPAREN = r'\)'
    t_SEM = r'\;'
    t_SHOW = r'\#show'
    t_EDGE = r'\#edge'
    t_PROJECT = r'\#project'
    t_HEURISTIC = r'\#heuristic'
    t_SHOWSIG = r'\#showsig'
    t_DOMAIN = r'\#domain'
    t_SLASH = r'\/'
    t_SUB = r'\-'
    t_SUM = r'\#sum'
    t_SUMP = r'\#sum+'
    t_SUPREMUM = r'\#sup'
    t_TRUE = r'\#true'
    t_BLOCK = r'\#program'
    t_VBAR = r'\|'
    t_WIF = r'\:\~'
    t_XOR = r'\^'
    t_COMMENT = r'\%.*'

    def t_IDENTIFIER(self,t):
        r'_*[a-z][\'a-zA-Z0-9_]*'
        t.type = self.reserved.get(t.value, 'IDENTIFIER')  # Check for reserved words
        return t

    def t_VARIABLE(self,t):
        r'_*[A-Z][\'a-zA-Z0-9_]*'
        t.type = self.reserved.get(t.value, 'VARIABLE')  # Check for reserved words
        return t

    def t_DIGIT(self,t):
        r'\d'
        try:
            t.value = int(t.value)
        except ValueError:
            print("integer value too large %d", t.value)
            t.value = 0

        return t

    # Ignored characters
    t_ignore = " \t"

    def t_newline(self,t):
        r'\n+'
        t.lexer.lineno += t.value.count("\n")

    def t_error(self,t):
        print("Illegal character '%s'" % t.value[0])
        t.lexer.skip(1)

    # Build the lexer

    # Parsing rules

    precedence = (
        ('left', 'DOTS'),
        ('left', 'XOR'),
        ('left', 'QUESTION'),
        ('left', 'AND'),
        ('left', 'ADD', 'SUB'),
        ('left', 'MUL', 'SLASH', 'MOD'),
        ('right', 'POW'),
    )

    def fill_t(self,t):
        filledt = ""
        for item in t:
            if type(item) is str:
                filledt += str(item)
            elif type(item) is int:
                filledt += str(item)
            elif type(item) is float:
                filledt += str(int(item))

        return filledt

    # {{{1 logic program and global definitions

    def p_start(self,t):
        'start : program'

    def p_program(self,t):
        '''program : program statement
                   | '''

    # Note: skip until the next "." in case of an error and switch back to normal lexing

    def p_statement(self,t):
        '''statement : statement_2
                    | statement_3
                    | statement_4
                    | statement_5
                    | statement_6
                    | statement_7
                    | statement_8
                    | statement_9
                    | statement_10
                    | statement_11
                    | statement_12
                    | statement_13
                    | statement_14
                    | statement_15'''

        if t[1][len(t[1]) - 1] == '\n':
            self.output += (t[1])
        else:
            self.output += (t[1] + '\n')

        self.sawDomainId = []
        self.ruleCounter += 1

    def p_identifier(self,t):
        'identifier : IDENTIFIER'
        t[0] = ' ' + self.fill_t(t)

    def p_vaiable(self,t):
        'variable : VARIABLE'

        for key, domaindicvar in self.g_domainDic.items():
            if t[1] == domaindicvar['var'].strip(' ') and t[1] not in self.sawDomainId:
                self.sawDomainId.append(t[1])

        t[0] = ' ' + self.fill_t(t)

    # {{{1 terms
    # {{{2 constterms are terms without variables and pooling operators

    def p_constterm(self,t):
        '''constterm : constterm XOR constterm
                    | constterm QUESTION constterm
                    | constterm AND constterm
                    | constterm ADD constterm
                    | constterm SUB constterm
                    | constterm MUL constterm
                    | constterm SLASH constterm
                    | constterm MOD constterm
                    | constterm POW constterm
                    | LPAREN RPAREN
                    | LPAREN COMMA RPAREN
                    | LPAREN consttermvec RPAREN
                    | LPAREN consttermvec COMMA RPAREN
                    | identifier LPAREN constargvec RPAREN
                    | AT identifier LPAREN constargvec RPAREN
                    | VBAR constterm VBAR
                    | identifier
                    | number
                    | STRING
                    | INFIMUM
                    | SUPREMUM'''

        if len(t) == 2 and type(t[1]) == int:
            t[0] = ' ' + self.fill_t(t)
        else:
            t[0] = self.fill_t(t)

    # {{{2 arguments lists for functions in constant terms

    def p_consttermvec(self,t):
        '''consttermvec : constterm
                        | consttermvec COMMA constterm'''
        t[0] = self.fill_t(t)

    def p_constargvec(self,t):
        '''constargvec : consttermvec
                        |
                        '''
        t[0] = self.fill_t(t)

    # {{{2 terms including variables

    def p_term(self,t):
        '''term : term DOTS term
                | term XOR term
                | term QUESTION term
                | term AND term
                | term ADD term
                | term SUB term
                | term MUL term
                | term SLASH term
                | term MOD term
                | term POW term
                | term WS
                | ADD term
                | SUB term
                | LPAREN tuplevec RPAREN
                | identifier LPAREN argvec RPAREN
                | AT identifier LPAREN argvec RPAREN
                | VBAR unaryargvec VBAR
                | variable
                | identifier
                | number
                | STRING
                | INFIMUM
                | SUPREMUM
                | ANONYMOUS'''

        if len(t) == 2 and type(t[1]) == int:
            t[0] = ' ' + self.fill_t(t)
        else:
            t[0] = self.fill_t(t)

    # {{{2 argument lists for unary operations

    def p_unaryargvec(self,t):
        '''unaryargvec : term
                        | unaryargvec SEM term'''
        t[0] = self.fill_t(t)

    # {{{2 argument lists for functions

    def p_ntermvec(self,t):
        '''ntermvec : term
                    | ntermvec COMMA term'''
        t[0] = self.fill_t(t)

    def p_termvec(self,t):
        '''termvec : ntermvec
                    |'''
        t[0] = self.fill_t(t)

    def p_tuple(self,t):
        '''tuple : ntermvec COMMA
                | ntermvec
                | COMMA
                | '''
        t[0] = self.fill_t(t)

    def p_tuplevec_sem(self,t):
        '''tuplevec_sem : tuple SEM
                        | tuplevec_sem tuple SEM '''
        t[0] = self.fill_t(t)

    def p_tuplevec(self,t):
        '''tuplevec : tuple
                    | tuplevec_sem tuple '''
        t[0] = self.fill_t(t)

    def p_argvec(self,t):
        '''argvec : termvec
                    | argvec SEM termvec'''
        t[0] = self.fill_t(t)

    def p_binaryargvec(self,t):
        '''binaryargvec : term COMMA term
                        | binaryargvec SEM term COMMA term'''
        t[0] = self.fill_t(t)

    # TODO: I might have to create tuples differently
    #       parse a tuple as a list of terms
    #       each term is either a tuple or a term -> which afterwards is turned into a pool!

    # {{{1 literals

    def p_cmp(self,t):
        '''cmp : GT
                | LT
                | GEQ
                | LEQ
                | EQ
                | NEQ'''
        t[0] = self.fill_t(t)

    def p_atom(self,t):
        '''atom : identifier
                | identifier LPAREN argvec RPAREN
                | SUB identifier
                | SUB identifier LPAREN argvec RPAREN'''

        t[0] = self.fill_t(t)

    def p_literal(self,t):
        '''literal : TRUE
                    | NOT TRUE
                    | NOT NOT TRUE
                    | FALSE
                    | NOT FALSE
                    | NOT NOT FALSE
                    | atom
                    | NOT atom
                    | NOT NOT atom
                    | term cmp term
                    | NOT term cmp term
                    | NOT NOT term cmp term
                    | csp_literal'''

        if len(t) >= 3 and t[1] == "not":
            t[1] = "not "
            if t[2] == "not":
                t[2] = "not "

        t[0] = self.fill_t(t)

    def p_csp_mul_term(self,t):
        '''csp_mul_term : CSP term CSP_MUL term
                        | term CSP_MUL CSP term
                        | CSP term
                        | term'''
        t[0] = self.fill_t(t)

    def p_csp_add_term(self,t):
        '''csp_add_term : csp_add_term CSP_ADD csp_mul_term
                        | csp_add_term CSP_SUB csp_mul_term
                        | csp_mul_term'''
        t[0] = self.fill_t(t)

    def p_csp_rel(self,t):
        '''csp_rel : CSP_GT
                    | CSP_LT
                    | CSP_GEQ
                    | CSP_LEQ
                    | CSP_EQ
                    | CSP_NEQ'''
        t[0] = self.fill_t(t)

    def p_csp_literal(self,t):
        '''csp_literal : csp_literal csp_rel csp_add_term
                        | csp_add_term  csp_rel csp_add_term'''
        t[0] = self.fill_t(t)

    # {{{1 aggregates

    # {{{2 auxiliary rules

    def p_nlitvec(self,t):
        '''nlitvec : literal
                    | nlitvec COMMA literal'''
        t[0] = self.fill_t(t)

    def p_litvec(self,t):
        '''litvec : nlitvec
                    | '''
        t[0] = self.fill_t(t)

    def p_optcondition(self,t):
        '''optcondition : COLON litvec
                         | '''
        t[0] = self.fill_t(t)

    def p_noptcondition(self,t):
        '''noptcondition : COLON nlitvec
                        | '''
        t[0] = self.fill_t(t)

    def p_aggregatefunction(self,t):
        '''aggregatefunction : SUM
                            | SUMP
                            | MIN
                            | MAX
                            | COUNT'''
        t[0] = self.fill_t(t)

    # {{{2 body aggregate elements

    def p_bodyaggrelem(self,t):
        '''bodyaggrelem : COLON litvec
                        | ntermvec optcondition'''
        t[0] = self.fill_t(t)

    def p_bodyaggrelemvec(self,t):
        '''bodyaggrelemvec : bodyaggrelem
                            | bodyaggrelemvec SEM bodyaggrelem'''
        t[0] = self.fill_t(t)

    # Note: alternative syntax (without weight)

    def p_altbodyaggrelem(self,t):
        'altbodyaggrelem : literal optcondition'
        t[0] = self.fill_t(t)

    def p_altbodyaggrelemvec(self,t):
        '''altbodyaggrelemvec : altbodyaggrelem
                              | altbodyaggrelemvec SEM altbodyaggrelem'''
        t[0] = self.fill_t(t)

    # {{{2 body aggregates

    def p_bodyaggregate(self,t):
        '''bodyaggregate : LBRACE RBRACE
                        | LBRACE altbodyaggrelemvec RBRACE
                        | aggregatefunction LBRACE RBRACE
                        | aggregatefunction LBRACE bodyaggrelemvec RBRACE'''
        t[0] = self.fill_t(t)

    def p_upper(self,t):
        '''upper : term
                | cmp term
                | '''
        t[0] = self.fill_t(t)

    def p_lubodyaggregate(self,t):
        '''lubodyaggregate : term bodyaggregate upper
                            | term cmp bodyaggregate upper
                            | bodyaggregate upper '''
        t[0] = self.fill_t(t)

    # {{{2 head aggregate elements

    def p_headaggrelemvec(self,t):
        '''headaggrelemvec : headaggrelemvec SEM termvec COLON literal optcondition
                            | termvec COLON literal optcondition '''
        t[0] = self.fill_t(t)

    def p_altheadaggrelemvec(self,t):
        '''altheadaggrelemvec : literal optcondition
                                | altheadaggrelemvec SEM literal optcondition '''
        t[0] = self.fill_t(t)

    # {{{2 head aggregates

    def p_headaggregate(self,t):
        '''headaggregate : aggregatefunction LBRACE RBRACE
                        | aggregatefunction LBRACE headaggrelemvec RBRACE
                        | LBRACE RBRACE
                        | LBRACE altheadaggrelemvec RBRACE '''
        t[0] = self.fill_t(t)

    def p_luheadaggregate(self,t):
        '''luheadaggregate : term headaggregate upper
                            | term cmp headaggregate upper
                            | headaggregate upper '''
        t[0] = self.fill_t(t)

    # {{{2 disjoint aggregate

    def p_ncspelemvec(self,t):
        '''ncspelemvec :  termvec COLON csp_add_term optcondition
                        | cspelemvec SEM termvec COLON csp_add_term optcondition'''
        t[0] = self.fill_t(t)

    def p_cspelemvec(self,t):
        '''cspelemvec : ncspelemvec
                        | '''
        t[0] = self.fill_t(t)

    def p_disjoint(self,t):
        '''disjoint : DISJOINT LBRACE cspelemvec RBRACE
                    | NOT DISJOINT LBRACE cspelemvec RBRACE
                    | NOT NOT DISJOINT LBRACE cspelemvec RBRACE '''
        t[0] = self.fill_t(t)

    # /}}}
    # {{{2 conjunctions

    def p_conjunction(self,t):
        'conjunction : literal COLON litvec'

        t[0] = self.fill_t(t)

    # }}}
    # {{{2 disjunctions

    def p_dsym(self,t):
        '''dsym : SEM
                | VBAR '''

    def p_disjunctionsep(self,t):
        '''disjunctionsep : disjunctionsep literal COMMA
                            | disjunctionsep literal noptcondition dsym
                            | '''
        t[0] = self.fill_t(t)

    # Note: for simplicity appending first condlit here
    def p_disjunction(self,t):
        '''disjunction : literal COMMA disjunctionsep literal noptcondition
                        | literal dsym disjunctionsep literal noptcondition
                        | literal  COLON nlitvec dsym disjunctionsep literal noptcondition
                        | literal COLON nlitvec '''
        t[0] = self.fill_t(t)

    # {{{1 statements
    # {{{2 rules

    def p_bodycomma(self,t):
        '''bodycomma : bodycomma literal COMMA
                    | bodycomma literal SEM
                    | bodycomma lubodyaggregate COMMA
                    | bodycomma lubodyaggregate SEM
                    | bodycomma NOT lubodyaggregate COMMA
                    | bodycomma NOT lubodyaggregate SEM
                    | bodycomma NOT NOT lubodyaggregate COMMA
                    | bodycomma NOT NOT lubodyaggregate SEM
                    | bodycomma conjunction SEM
                    | bodycomma disjoint SEM
                    |'''

        if len(t) >= 2 and t[2] == "not":
            t[2] = "not "

        t[0] = self.fill_t(t)

    def p_bodydot(self,t):
        '''bodydot : bodycomma literal DOT
                    | bodycomma lubodyaggregate DOT
                    | bodycomma NOT lubodyaggregate DOT
                    | bodycomma NOT NOT lubodyaggregate DOT
                    | bodycomma conjunction DOT
                    | bodycomma disjoint DOT '''

        if len(t) >= 2 and t[2] == "not":
            t[2] = "not "

        t[0] = self.fill_t(t)

    def p_bodyconddot(self,t):
        '''bodyconddot : DOT
                        | COLON DOT
                        | COLON bodydot '''
        t[0] = self.fill_t(t)

    def p_head(self,t):
        '''head : literal
                | disjunction
                | luheadaggregate '''
        t[0] = self.fill_t(t)

    # Next is the soft rule
    def p_statement_14(self,t):
        '''statement_14 : head DOT
                        | head IF DOT
                        | head IF bodydot
                        | IF bodydot
                        | IF DOT'''


        self.domainstr = ""
        if len(self.sawDomainId) != 0:
            for key, value in self.g_domainDic.items():
                for item in self.sawDomainId:
                    if item in value['var'].strip(' '):
                        self.domainstr += ' ,' + value['id'] + '(' + value['var'] + ')'

        if self.domainstr != "":
            if len(t) == 3 and t[2] == '.':
                t[0] = self.fill_t(t)[:-1] + ' :- ' + self.domainstr[2:] + '.'
            else:
                t[0] = self.fill_t(t)[:-1] + self.domainstr + '.'
        else:
            t[0] = self.fill_t(t)

    def p_statement_15(self,t):
        '''statement_15 : DOMAIN identifier LPAREN argvec RPAREN DOT'''

        self.g_domainDic[str(self.domainCounter)] = {
            'id': t[2],
            'var': t[4]
        }

        self.domainCounter += 1
        t[0] = '%' + self.fill_t(t)

    # {{{2 CSP

    def p_statement_2(self,t):
        '''statement_2 : disjoint IF bodydot
                        | disjoint IF DOT
                        | disjoint DOT'''
        t[0] = self.fill_t(t)

    # {{{2 optimization

    def p_optimizetuple(self,t):
        '''optimizetuple : COMMA ntermvec
                         |'''
        t[0] = self.fill_t(t)

    def p_optimizeweight(self,t):
        '''optimizeweight : term AT term
                          | term'''
        t[0] = self.fill_t(t)

    def p_optimizelitvec(self,t):
        '''optimizelitvec : literal
                          | optimizelitvec COMMA literal'''
        t[0] = self.fill_t(t)

    def p_optimizecond(self,t):
        '''optimizecond : COLON optimizelitvec
                        | COLON
                        | '''
        t[0] = self.fill_t(t)

    def p_statement_3(self,t):
        '''statement_3 : WIF bodydot LBRACK optimizeweight optimizetuple RBRACK
                       | WIF DOT LBRACK optimizeweight optimizetuple RBRACK'''
        t[0] = self.fill_t(t)

    def p_maxelemlist(self,t):
        '''maxelemlist : optimizeweight optimizetuple optimizecond
                       | maxelemlist SEM optimizeweight optimizetuple optimizecond'''
        t[0] = self.fill_t(t)

    def p_minelemlist(self,t):
        '''minelemlist : optimizeweight optimizetuple optimizecond
                       | minelemlist SEM optimizeweight optimizetuple optimizecond '''
        t[0] = self.fill_t(t)

    def p_statement_4(self,t):
        '''statement_4 : MINIMIZE LBRACE RBRACE DOT
                        | MAXIMIZE LBRACE RBRACE DOT
                        | MINIMIZE LBRACE minelemlist RBRACE DOT
                        | MAXIMIZE LBRACE maxelemlist RBRACE DOT'''
        t[0] = self.fill_t(t)

    # {{{2 visibility

    def p_statement_5(self,t):
        '''statement_5 : SHOWSIG identifier SLASH number DOT
                        | SHOWSIG SUB identifier SLASH number DOT
                        | SHOW DOT
                        | SHOW term COLON bodydot
                        | SHOW term DOT
                        | SHOWSIG CSP identifier SLASH number DOT
                        | SHOW CSP term COLON bodydot
                        | SHOW CSP term DOT'''
        t[0] = self.fill_t(t)

    # {{{2 acyclicity

    def p_statement_6(self,t):
        'statement_6 : EDGE LPAREN binaryargvec RPAREN bodyconddot'
        t[0] = self.fill_t(t)

    # {{{2 heuristic

    def p_statement_7(self,t):
        '''statement_7 : HEURISTIC atom bodyconddot LBRACK term AT term COMMA term RBRACK
                        | HEURISTIC atom bodyconddot LBRACK term COMMA term RBRACK'''
        t[0] = self.fill_t(t)

    # {{{2 project

    def p_statement_8(self,t):
        '''statement_8 : PROJECT identifier SLASH number DOT
                        | PROJECT SUB identifier SLASH number DOT
                        | PROJECT atom bodyconddot'''
        t[0] = self.fill_t(t)

    # {{{2 constants

    '''def p_define(t):
        'define : identifier EQ constterm'
        t[0] = self.fill_t(t)'''

    def p_statement_9(self,t):
        'statement_9 : CONST identifier EQ constterm DOT'
        t[0] = self.fill_t(t)

    def p_statement_10(self,t):
        '''statement_10 : INCLUDE STRING DOT
                        | INCLUDE LT identifier GT DOT'''
        t[0] = self.fill_t(t)

    # {{{2 blocks

    def p_nidlist(self,t):
        '''nidlist : nidlist COMMA identifier
                   | identifier '''
        t[0] = self.fill_t(t)

    def p_idlist(self,t):
        '''idlist :
                  | nidlist '''
        t[0] = self.fill_t(t)

    def p_statement_11(self,t):
        '''statement_11 : BLOCK identifier LPAREN idlist RPAREN DOT
                        | BLOCK identifier DOT'''
        t[0] = self.fill_t(t)

    # {{{2 external

    def p_statement_12(self,t):
        '''statement_12 : EXTERNAL atom COLON bodydot
                        | EXTERNAL atom COLON DOT
                        | EXTERNAL atom DOT'''
        t[0] = self.fill_t(t)

    def p_statement_13(self,t):
        'statement_13 : COMMENT'
        t[0] = self.fill_t(t)

    def p_digit_number(self,t):
        '''number : DIGIT number
                  | DIGIT'''

        t[0] = self.fill_t(t)

    def p_error(self,t):
        if type(t) is 'NoneType':
            print("Syntax error")
        else:
            print("Syntax error at '%s'" % t.value)
        print("Type of t is: " + t.type)
        print("At rule line number: " + str(self.ruleCounter))
        print("next token: " + str(self.parser.token()))

    def build(self, **kwargs):
        self.lexer = domain_lex.lex(module=self, **kwargs)



    def __init__(self,content):

        self.domainCounter = 0
        self.sawDomainId = []
        self.output = ""
        self.g_domainDic = {}
        self.ruleCounter = 1
        self.content = content
        self.build()
        self.parser = yacc.yacc(module=self,start='start', debug=False, write_tables=False)

    def parseToRemoveDomain(self):
        self.parser.parse(self.content,self.lexer)
        return self.output