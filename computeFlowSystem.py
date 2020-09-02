import numpy as np
import math
import sys
import Tensor
from grad import gradSystem
import matrixSystem
from functools import reduce
import copy

class ComputeFlowSystem(object):
    # consts
    # 通过正反向检查，保证一定可以遍历计算图中所有结点
    compute_flows={} # compute_flows 里存储的是实体.可以根据一个结果找到多个父级对象,对应到计算图的BP，反向传递
    lineage={} # lineage 里存储的是每一个变量与其进行计算过的所有变量的hash值，之后可以通过自连接、遍历找到结构树,相当于正向查找。

    #labels为预留参数
    @staticmethod
    def createVariable(vectors,flow_id=None,require_grad=None,labels=None,sync=None):
        result=Tensor.tensor(vectors,flow_id,require_grad,labels)
        if sync!=None:
            for variable,func,args,kwargs in sync:
                if require_grad: 
                    temp=variable.step_result
                    variable.step_result=True
                    ComputeFlowSystem.setLineAge(variable,func,args,kwargs,result)
                    variable.step_result=temp
        return result

    @staticmethod
    def getBackwards(y,x):#得到反向变量结构树
        backwards = []
        cached=[]
        # print(x.flow_id==y.flow_id)
        if x.flow_id==y.flow_id: # 如果处于同一分支才进行求导
            def loop(xn,backward):
                if not isinstance(xn[0],np.ndarray):
                    backward.append(xn)
                if xn[0].id==y.id:
                    backwards.append(backward)
                else:
                    for j in xn:
                        if j.id in ComputeFlowSystem.lineage[x.flow_id].keys() and j.id not in cached:
                            if hasattr(j,"func") and j.func.__name__ =='ldot':  # 假如cached后对原来的求导有影响，暂时加个强制判断
                                cached.append(j.id)
                            for i in ComputeFlowSystem.lineage[x.flow_id][j.id]:
                                loop(ComputeFlowSystem.compute_flows[i][j.id], copy.copy(backward))
            loop([x],[])
        return backwards

    @staticmethod
    def setLineAge(variable, func, args, kwargs, result):#创建结构树，类似于SPARK RDD的关系
        if variable.step_result:
            if result.id not in ComputeFlowSystem.compute_flows.keys():
                ComputeFlowSystem.compute_flows[result.id]=dict({})
            if variable.id not in ComputeFlowSystem.compute_flows[result.id].keys():
                ComputeFlowSystem.compute_flows[result.id][variable.id] = []
            ComputeFlowSystem.compute_flows[result.id][variable.id].append(
                    gradSystem.GradSystem.createGrad(variable, func, args, kwargs, result))
            if variable.flow_id not in ComputeFlowSystem.lineage.keys():
                ComputeFlowSystem.lineage[variable.flow_id] = dict({})
            if variable.id not in ComputeFlowSystem.lineage[variable.flow_id].keys():
                ComputeFlowSystem.lineage[variable.flow_id][variable.id] = []
                # print(variable.flow_id, variable.id)
            ComputeFlowSystem.lineage[variable.flow_id][variable.id].append(result.id)

    @staticmethod
    def insertVariable(d_func=None,from_func="main"):
        def wapper(func):
            #如果有现成的导函数，就直接使用
            if d_func!=None:
                gradSystem.GradSystem.d_functions[func]=d_func
            def inner_wapper(variable,*args,**kwargs):
                if isinstance(variable, Tensor.TensorVar):
                    Tensor.TensorVar.from_func=from_func
                    result=func(variable,*args,**kwargs)
                    Tensor.TensorVar.from_func="main"
                    if variable.step_result: 
                        # if isinstance(result,dict):
                        #     ComputeFlowSystem.setLineAge(result["variable"],func,
                        #                                   result["args"],result["kwargs"],result["result"])
                        #     return result["result"]
                        # else:
                            ComputeFlowSystem.setLineAge(variable,func,list(args),kwargs,result)
                    return result
                else:
                    raise RuntimeError("variable must be type of Tensor.tensor")
            return inner_wapper
        return wapper


    @staticmethod
    def grad(y,x=None,diagonal=False,default_func=False): #导数(得到导函数值）

        if type(x)==type(None):
            x=ComputeFlowSystem.flows[list(ComputeFlowSystem.lineage[y.flow_id].keys())[0]][0]

        if x.id==y.id:
            # 对自身求导
            return [1]
    
        grad_lines=ComputeFlowSystem.getBackwards(y,x)
        if len(grad_lines)==0:
            # 没有derivation关系
            return [1]
        # 链式法则
        results=[reduce(lambda x,y:(np.array(x)*y),
                        [reduce(lambda x,y:list(x)+list(y),[gradSystem.GradSystem.requireGrad(grad,default_func=default_func) for grad in grads])
                                                                for grads in grad_line]) for grad_line in grad_lines]

        #是否忽略线性函数变换只返回对角矩阵
        if not diagonal:
            return np.squeeze(results)

        #找出最大形式
        diagonal_grads=[matrixSystem.MatrixSystem.getDiagonal(r[0],sorted([np.shape(g[0].get_variable()) for g in r[1]], key=lambda s: len(s))[0]) for r in zip(results,grad_lines)]
        return np.around(reduce(lambda x, y:(np.array(x)+y).tolist() ,diagonal_grads),5).tolist()

    def plot(self):#画图
        pass