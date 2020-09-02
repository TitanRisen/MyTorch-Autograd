import numpy as np
from . import gradController
from Toolkit import *
class GradSystem(object):

    grads=[]
    d_functions=dict({})

    @staticmethod
    def createGrad(variable, func, args, kwargs, result):
        grad=gradController.Grad(variable, func, args, kwargs, result)
        return grad


    @staticmethod
    def isGrad(o):
        return isinstance(o, gradController.Grad)

    @staticmethod
    def requireGrad(grad,fullGard=False,default_func=False):
        '''
            求导核心部分，生成jacobi / Hessian 矩阵
        '''
        cased=GradSystem.caseFunc(grad)

        if type(cased)!=bool :
            grad.cased=True
            return cased
        if default_func: #有导函数直接用
            default_grad=GradSystem.defaultFunc(grad)
            if type(default_grad)!=bool:
                return default_grad
        def dx(elem, index, va):
            '''
            参考torch的求导方式，取邻近点算斜率
            '''
            grad_variable = grad.get_variable()

            v = np.array(grad_variable, dtype='float64')
            v[index] += grad.d
            grad_variable.set(v.tolist())
            if not fullGard:
                df_divide_dx = [e for e in forEveryElemWithIndex(grad.call(grad_variable), lambda e, i, v: e / grad.d)]
                if len(df_divide_dx) == 1:
                    return df_divide_dx[0]
                return df_divide_dx
            df_divide_dx = np.array(grad.call(grad_variable), dtype='float64')[index] / grad.d
            return df_divide_dx
        return forEveryElemWithIndex(grad.variable, dx)

    @staticmethod
    def defaultFunc(gard):
        if gard.func in GradSystem.d_functions:
            return GradSystem.d_functions[gard.func](gard.variable,*gard.args,**gard.kwargs)
        return False

    @staticmethod
    def caseFunc(grad):
        if "__mul__" in str(grad.func) or "__rmul__" in str(grad.func):
            return grad.args
        if "__truediv__" in str(grad.func):
            return (1 / np.array(grad.args)).tolist()
        if "__rtruediv__" in str(grad.func):
            return -grad.args / grad.variable ** 2
        if "ldot" in str(grad.func):
            return grad.args
        if "__radd__" in str(grad.func):
            return 1
        if "__rsub__" in str(grad.func):
            return -1
        
        return False