import sys
from rpython import conftest

from mnist import main



class o:
    view = False
    viewloops = True
conftest.option = o

from rpython.rlib.nonconst import NonConstant
from rpython.rlib import jit
from rpython.jit.metainterp.test.test_ajit import LLJitMixin

from myxor2 import main

class TestLLtype(LLJitMixin):
    def test_main(self):
        def f():
            return main([])
        self.meta_interp(f, [], listcomp=True, listops=True, backendopt=True, inline=True)
