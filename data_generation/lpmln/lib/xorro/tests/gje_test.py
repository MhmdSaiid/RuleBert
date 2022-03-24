"""
Gauss-Jordan Tests Suite
"""
from xorro import gje

def cols_state_to_matrix(state):
    ## Parse columns state to matrix
    return gje.columns_state_to_matrix(state)

def get_clause(m,lits):
    ## Deduce clause after GJE
    return gje.deduce_clause(m,lits)

def xor_columns(col,parity):
    ## XOR parity column with parity column
    return gje.xor_columns(col,parity)

def swap_row(m,i,j):
    ## Swap Rows m[i] with m[j]
    return gje.swap(m,i,j)

def xor_row(m,i,j):
    ## XOR Rows m[i] with m[j]
    return gje.xor(m,i,j)
    
def remove_rows_zeros(m):
    ## Remove rows with all zeros including the augmented column
    matrix = gje.remove_rows_zeros(m)
    return matrix

def check_sat(m):
    ## Check SAT
    return gje.check_sat(m)
    
def solve_gje(m, show):
    ## If there are more than unary xors perform GJE
    if len(m[0]) > 2:
        m = gje.remove_rows_zeros(m)
        m = gje.perform_gauss_jordan_elimination(m, show)
    return m


"""
Gauss-Jordan Exclusive Tests
Parse the columns state to a binary matrix and return the list of literals
"""
def test_columns_state_to_matrix(self):
    self.assertEqual(cols_state_to_matrix(
    {'parity': [0, 1, 1, 0, 0], 2: [1, 0, 0, 1, 0], 3: [0, 0, 0, 0, 1], 4: [1, 1, 0, 0, 0], 5: [0, 1, 0, 0, 0], 6: [1, 1, 0, 0, 0], 7: [0, 0, 1, 0, 1], 8: [0, 0, 1, 0, 0], 9: [0, 0, 0, 1, 0], 10: [0, 0, 0, 1, 0]}),
                     ([[1,0,1,0,1,0,0,0,0,0],
                       [0,0,1,1,1,0,0,0,0,1],
                       [0,0,0,0,0,1,1,0,0,1],
                       [1,0,0,0,0,0,0,1,1,0],
                       [0,1,0,0,0,1,0,0,0,0]],[2,3,4,5,6,7,8,9,10]))

    
"""
Deduce clause after Gauss-Jordan Elimination
"""
def test_get_clauses(self):
    self.assertEqual(get_clause([[1, 0, 0, 0],
                                 [0, 1, 1, 0],
                                 [0, 0, 0, 0]], [2,3,4]), [-2])

    self.assertEqual(get_clause([[1, 0, 0, 0],
                                 [0, 1, 0, 0],
                                 [0, 0, 1, 1]], [2,3,4]), [-2,-3,4])

    self.assertEqual(get_clause([[1, 0, 1, 1, 0, 1],
                                 [0, 1, 1, 0, 0, 0],
                                 [1, 0, 1, 1, 1, 0]], [2,3,4,5,6]), [])

"""
XOR a single column with Parity column Tests
"""
def test_xor_columns(self):
    self.assertEqual(xor_columns([1, 0],[1, 0]),[0, 0])

    self.assertEqual(xor_columns([0, 0, 0, 0, 0],[1, 1, 1, 1, 1]),[1, 1, 1, 1, 1])

    self.assertEqual(xor_columns([0, 1, 0, 1],[1, 0, 1, 0]),[1, 1, 1, 1])

    
"""
Swap Rows Tests
"""
def test_swap_rows(self):
    self.assertEqual(swap_row([[1, 0, 1, 1, 1, 1],
                               [1, 1, 0, 1, 0, 1],
                               [1, 0, 0, 0, 0, 1]], 1, 2),[[1, 0, 1, 1, 1, 1],
                                                           [1, 0, 0, 0, 0, 1],
                                                           [1, 1, 0, 1, 0, 1]])

    self.assertEqual(swap_row([[0, 0],
                               [1, 1]], 1, 0),[[1, 1],
                                               [0, 0]])

    self.assertEqual(swap_row([[0, 1],
                               [1, 0]], 1, 0),[[1, 0],
                                               [0, 1]])

"""
XOR Rows Tests
"""
def test_xor_rows(self):
    self.assertEqual(xor_row([[1, 0],
                              [1, 1],
                              [1, 0]], 0, 1),[[1, 0],
                                              [0, 1],
                                              [1, 0]])

    self.assertEqual(xor_row([[0, 0],
                              [1, 1]], 1, 0),[[1, 1],
                                              [1, 1]])

    self.assertEqual(xor_row([[0, 0],
                              [0, 0]], 1, 0),[[0, 0],
                                              [0, 0]])

""" 
Pre GJE... Remove Rows if they are all zeros
"""
## Remove Rows full of Zeros 
def test_remove_zeros(self):
    self.assertEqual(remove_rows_zeros([[1, 0, 1, 0],
                                        [1, 1, 1, 0],
                                        [0, 1, 0, 1],
                                        [0, 0, 0, 0],
                                        [0, 0, 0, 0]]),
                     [[1, 0, 1, 0],
                      [1, 1, 1, 0],
                      [0, 1, 0, 1]])

    self.assertEqual(remove_rows_zeros([[1, 0, 0],
                                        [0, 1, 1],
                                        [0, 0, 1],
                                        [0, 0, 0]]),
                     [[1, 0, 0],
                      [0, 1, 1],
                      [0, 0, 1]])

    self.assertEqual(remove_rows_zeros([[0, 1, 1],
                                        [1, 0, 0],
                                        [0, 0, 0]]),
                     [[0, 1, 1],
                      [1, 0, 0]])


""" 
Check Satisfiability/Conflict wrt the augmented column. 
Return True if conflict (It must exist an empty odd equation)
"""
## Check SATISFIABILITY
def test_check_sat(self):
    self.assertEqual(check_sat([[1, 0, 1, 0],
                                 [1, 1, 1, 0],
                                 [0, 1, 0, 1],
                                 [0, 0, 0, 1],
                                 [0, 0, 0, 0]]),True)

    self.assertEqual(check_sat([[1, 0, 0],
                                [0, 1, 1],
                                [1, 0, 1],
                                [1, 1, 0]]),False)

    self.assertEqual(check_sat([[1, 0, 1],
                                [0, 1, 0],
                                [0, 0, 1]]),True)

    self.assertEqual(check_sat([[1, 0, 0, 0, 0, 0],
                                [0, 1, 0, 0, 0, 1],
                                [0, 0, 1, 0, 0, 1],
                                [0, 0, 0, 1, 0, 0],
                                [0, 0, 0, 0, 1, 1]]),False)


"""
Gauss-Jordan Elimination Tests

The second parameter in the solve function is a flag.
If True, it will display the GJ Elimination Procedure

"""

## No GJE due matrix size. Return the same matrix to check SAT
def test_no_gje(self):
    self.assertEqual(solve_gje([[1, 0],
                                [1, 1],
                                [1, 0]],False),
                     [[1, 0],
                      [1, 1],
                      [1, 0]])

    self.assertEqual(solve_gje([[1, 0],
                                [0, 1]],False),
                     [[1, 0],
                      [0, 1]])


## More Columns than Rows
def test_more_cols(self):
    self.assertEqual(solve_gje([[0, 1, 1, 0, 0],
                                [0, 1, 1, 0, 0],
                                [1, 0, 0, 1, 0]],False),
                     [[1, 0, 0, 1, 0],
                      [0, 1, 1, 0, 0],
                      [0, 0, 0, 0, 0]])

    self.assertEqual(solve_gje([[0, 1, 1, 0],
                                [0, 1, 1, 0],
                                [1, 0, 0, 0]],False),
                     [[1, 0, 0, 0],
                      [0, 1, 1, 0],
                      [0, 0, 0, 0]])

    self.assertEqual(solve_gje([[1, 0, 1, 0, 0, 0, 0, 0],
                                [1, 1, 1, 0, 0, 0, 0, 1],
                                [0, 0, 0, 1, 1, 0, 0, 1],
                                [0, 0, 0, 0, 0, 1, 1, 0],
                                [0, 0, 0, 1, 0, 0, 0, 0]],False),
                     [[1, 0, 1, 0, 0, 0, 0, 0],
                      [0, 1, 0, 0, 0, 0, 0, 1],
                      [0, 0, 0, 1, 0, 0, 0, 0],
                      [0, 0, 0, 0, 1, 0, 0, 1],
                      [0, 0, 0, 0, 0, 1, 1, 0]])

    self.assertEqual(solve_gje([[0, 1, 0, 0, 0, 0, 0, 1],
                                [0, 1, 1, 0, 0, 0, 0, 0],
                                [0, 0, 0, 1, 1, 0, 0, 1],
                                [0, 0, 0, 0, 0, 1, 1, 0],
                                [0, 1, 0, 0, 0, 0, 1, 0],
                                [1, 0, 0, 1, 0, 0, 0, 0]],False),
                     [[1, 0, 0, 0, 1, 0, 0, 1],
                      [0, 1, 0, 0, 0, 0, 0, 1],
                      [0, 0, 1, 0, 0, 0, 0, 1],
                      [0, 0, 0, 1, 1, 0, 0, 1],
                      [0, 0, 0, 0, 0, 1, 0, 1],
                      [0, 0, 0, 0, 0, 0, 1, 1]])

    self.assertEqual(solve_gje([[1, 0, 1, 0, 1, 1, 0, 0],
                                [1, 1, 1, 0, 0, 0, 1, 1],
                                [0, 0, 1, 0, 1, 0, 0, 1],
                                [0, 1, 0, 1, 0, 1, 1, 0],
                                [0, 0, 0, 1, 0, 0, 0, 0]],False),
                     [[1, 0, 0, 0, 0, 1, 0, 1],
                      [0, 1, 0, 0, 0, 1, 1, 0],
                      [0, 0, 1, 0, 0, 0, 0, 0],
                      [0, 0, 0, 1, 0, 0, 0, 0],
                      [0, 0, 0, 0, 1, 0, 0, 1]])

## Square Matrix
def test_square(self):
    self.assertEqual(solve_gje([[1, 0, 1, 0, 1, 0],
                                [1, 1, 1, 0, 0, 1],
                                [0, 0, 1, 0, 1, 1],
                                [0, 1, 0, 1, 0, 0],
                                [0, 0, 0, 1, 0, 0]],False),
                     [[1, 0, 0, 0, 0, 1],
                      [0, 1, 0, 0, 0, 0],
                      [0, 0, 1, 0, 0, 0],
                      [0, 0, 0, 1, 0, 0],
                      [0, 0, 0, 0, 1, 1]])

    self.assertEqual(solve_gje([[1, 0, 1, 1, 1],
                                [1, 0, 1, 0, 0],
                                [0, 1, 0, 0, 1],
                                [0, 0, 1, 1, 0]],False),
                     [[1, 0, 0, 0, 1],
                      [0, 1, 0, 0, 1],
                      [0, 0, 1, 0, 1],
                      [0, 0, 0, 1, 1]])

    self.assertEqual(solve_gje([[1, 1, 1, 1],
                                [1, 0, 1, 0],
                                [0, 0, 1, 0]],False),
                     [[1, 0, 0, 0],
                      [0, 1, 0, 1],
                      [0, 0, 1, 0]])

    self.assertEqual(solve_gje([[0, 0, 1, 1, 1, 0],
                                [0, 1, 1, 1, 0, 1],
                                [1, 0, 1, 1, 1, 1],
                                [0, 1, 0, 1, 0, 0],
                                [1, 0, 0, 1, 0, 1]],False),
                     [[1, 0, 0, 0, 0, 1],
                      [0, 1, 0, 0, 0, 0],
                      [0, 0, 1, 0, 0, 1],
                      [0, 0, 0, 1, 0, 0],
                      [0, 0, 0, 0, 1, 1]])

    self.assertEqual(solve_gje([[1, 1, 1],
                                [1, 0, 1]],False),
                     [[1, 0, 1],
                      [0, 1, 0]])


## More Rows than Columns
def test_more_rows(self):
    self.assertEqual(solve_gje([[1, 0, 1, 0],
                                [1, 1, 1, 0],
                                [0, 1, 0, 1],
                                [0, 0, 1, 0],
                                [0, 1, 0, 1]],False),
                     [[1, 0, 0, 0],
                      [0, 1, 0, 0],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1],
                      [0, 0, 0, 1]])

    self.assertEqual(solve_gje([[0, 1, 0],
                                [0, 1, 1],
                                [1, 0, 0],
                                [1, 1, 0]],False),
                     [[1, 0, 0],
                      [0, 1, 1],
                      [0, 0, 1],
                      [0, 0, 1]])

    self.assertEqual(solve_gje([[0, 1, 1],
                                [1, 0, 0],
                                [0, 0, 0]],False),
                     [[1, 0, 0],
                      [0, 1, 1]])



