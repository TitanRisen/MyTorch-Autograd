import copy
import numpy
class Grad():
    d=10**-6 # 取一个临近点的邻域范围
    cased = False
    
    def __init__(self, variable, func, args, kwargs,result):
        self.variable =variable
        self.func = func
        self.args = args
        self.kwargs = kwargs
        self.result=result
        self.id = result.id

    def get_variable(self):
        return copy.copy(self.variable)

    def call(self, variable):
        f=self.func(variable,*self.args,**self.kwargs)
        #df=f(dx)-f(x)
        f.set((numpy.array(f.tolist())-self.result.tolist()).tolist())
        return f