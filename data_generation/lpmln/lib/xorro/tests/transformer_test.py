import unittest
import sys
import xorro
from xorro import transformer
import clingo
from textwrap import dedent
from . import gje_test

class TestCase(unittest.TestCase):
    def assertRaisesRegex(self, *args, **kwargs):
        return (self.assertRaisesRegexp(*args, **kwargs)
            if sys.version_info[0] < 3
            else unittest.TestCase.assertRaisesRegex(self, *args, **kwargs))
    
def solve(s, mode):
    messages = []
    prg = clingo.Control(logger=lambda c, m: messages.append(m))
    with prg.builder() as b:
        transformer.transform([s], b.add)
    prg.ground([("base", [])])

    xorro.translate(mode, prg, 0.0)

    prg.configuration.solve.models = 0

    models = []
    with prg.solve(yield_=True) as it:
        for m in it:
            models.append([str(sym) for sym in m.symbols(atoms=True) if not sym.name.startswith("__")])
            models[-1].sort()

    models.sort()
    return models


class TestProgramTransformer(TestCase):

    modes = ["count", "list", "tree", "countp", "up", "gje", "reason-gje"]

    def test_trivial(self):
        for mode in TestProgramTransformer.modes:
            self.assertEqual(solve("", mode), [[]])

    def test_empty_even(self):
        for mode in TestProgramTransformer.modes:
            self.assertEqual(solve("&even{ }.", mode), [[]])

    def test_empty_odd(self):
        for mode in TestProgramTransformer.modes:
            self.assertEqual(solve("&odd{ }.", mode), [])

    def test_basic(self):
        for mode in TestProgramTransformer.modes:
            self.assertEqual(solve("{a;b;c}. &even{ a:a;b:b;c:c }.", mode), [[], ["a", "b"], ['a', 'c'], ['b', 'c']])
            self.assertEqual(solve("{a}. &even{ a:a }.", mode), [[]])
            self.assertEqual(solve("{a}. &odd{ a:a }.", mode), [["a"]])

    def test_negative(self):
        for mode in TestProgramTransformer.modes:
            self.assertEqual(solve("{a;b;c}. &even{ a:a;b:not b;c:c }.", mode), [['a'], ['a', 'b', 'c'], ['b'], ['c']])
            self.assertEqual(solve("{a;c}. b :- a. &even{ a:a;b:b;c:c }.", mode), [[], ['a', 'b']])
            self.assertEqual(solve("{a;c}. b :- not a. &even{ a:a;b:b;c:c }.", mode), [['a', 'c'], ['b', 'c']])

    def test_xor_and_facts(self):
        for mode in TestProgramTransformer.modes:
            self.assertEqual(solve("{a;b;c}. &even{ a:a;b:b;c:c }. a.", mode), [["a", "b"], ['a', 'c']])

    def test_complex(self):
        prg = dedent("""\
            {p(1..10)}.
            &even{ X : p(X), X=1..7 }.
            &odd{  X : p(X), X=3..10 }.
            &odd{  X : not p(X), X=1..3; X : p(X), X=7..10 }.
            &odd{  X : p(X), X=1..4; X : not p(X), X=6..10 }.
            &even{  X : p(X), X=4..9; X : not p(X), X=7..8 }.
            """)
        models = solve(prg, "count")
        for mode in TestProgramTransformer.modes:
            if mode != "count":
                self.assertEqual(solve(prg, mode), models)


    ## Gauss-Jordan Elimination Tests
    def test_columns_state_to_matrix(self):
        gje_test.test_columns_state_to_matrix(self)

    def test_get_clause(self):
        gje_test.test_get_clauses(self)

    def test_xor_columns(self):
        gje_test.test_xor_columns(self)

    def test_swap_rows(self):
        gje_test.test_swap_rows(self)

    def test_xor_rows(self):
        gje_test.test_xor_rows(self)

    def test_remove_zeros(self):
        gje_test.test_remove_zeros(self)

    def test_check_sat(self):
        gje_test.test_check_sat(self)

    def test_no_gje(self):
        gje_test.test_no_gje(self)

    def test_more_columns(self):
        gje_test.test_more_cols(self)

    def test_square_matrix(self):
        gje_test.test_square(self)

    def test_more_rows(self):
        gje_test.test_more_rows(self)

