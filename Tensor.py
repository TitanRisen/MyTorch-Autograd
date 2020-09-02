from computeFlowSystem import ComputeFlowSystem as CFS
from Toolkit import *
import pickle
import math
from numpy import matlib
import numpy as np
import random
from functools import reduce
import copy


undo_funcs=["dot"] # 不另外占用内存的函数类型

def tensor(vectors=None, flow_id=None, require_grad=False,label=[],dtype=np.float,offset=0,format_check=False):
    '''
    # 创建一个tensor/variable
        vectors: list数组或numpy数组
        flow_id: 分支id
        require_grad: 是否需要中间求导
        lable: 标签，预留参数
        dtype,offset： 参考numpy用法
        format_check: 控制格式检查
    '''
    # print(vectors)
    #vectors=TensorVar.sequece(vectors,full=np.nan)
    # print(vectors,1)

    if TensorVar.from_func in undo_funcs:
        return np.asarray(vectors)
    if type(vectors) == type(None):
        return TensorVar
    if flow_id==None and format_check:
        
        vectors=TensorVar.Sequece(vectors,full=np.nan)

        if not hasattr(vectors, "__iter__") :
            vectors = [vectors]

    v=TensorVar(np.shape(vectors), dtype, np.array(vectors,dtype=dtype), offset)

    return v.init(flow_id, require_grad, label)


class TensorVar(np.ndarray):
    '''
    类似torch中的Tensor,继承np.ndarray

    '''
    # 用类变量声明方便dump
    labels = []
    flow_id = 0
    step_result = False
    from_func="main"
    # id = None

    def params(self):
        return self.flow_id,self.step_result,self.labels
    def init(self, flow_id=None, step_result=False,label=[]):
        '''
        #该子类的事例创建函数,不overwrite np.ndarray本身的构造函数
        flow_id: 计算图里的分支id
        step_result：是否是中间结果
        label：标签（预留参数）
        from_func: 来源函数
        '''
        #super().__init__(vectors)
        self.labels = label
        self.all_labels = {"y": label}
        self.flow_id =id(self) if flow_id==None else flow_id
        self.from_func=TensorVar.from_func
        self.step_result = step_result
        self.id = id(self)
        return self

    # 提交variable到计算图中
    def submitVariable(self,variable,sync=None,labels=None,from_func="main"):
        '''
        sync:如果涉及多变量运算，则表示同步求导的参数列表，即N个变量相互运算，会同时调用N个变量的求导方法然后合并。
        '''
        if not labels:labels=self.labels
        if sync == None:
            return tensor(variable,self.flow_id,self.step_result,labels)
        return CFS.createVariable(variable,self.flow_id,self.step_result,labels,sync)

    def __str__(self):
        try:
            #return str(forEveryElemWithIndex(self.tolist(),lambda e:None if e!=e else e,clear_null=True))
            return str(self.tolist())
        except :
            return super().__str__()
    def reduce(self,func):
        from functools import reduce
        return self.submitVariable(reduce(func,self))

    def set_param(self,k,v):
        self.__setattr__(k,v)
        return self

    def set_compflow(self,other):
        try:
            self.flow_id = other.flow_id
        except:
            raise RuntimeError("param must be type of Tensor.tensor")

    #列出常用函数的导函数,x就是self
    def d_relu(x):
        x[x<=0]= 0
        x[x>0] = 1
        return x

    def d_softMax(x):
        return [1]

    def d_sigmoid(x):
        return tensor(forEveryElem(lambda elem: 1. / (1. + math.exp(-elem)) * (1. - 1. / (1. + math.exp(-elem))), x))

    def d_log(x,base=np.e):
        return tensor(forEveryElem(lambda elem: 1./( elem * np.log(base)),x))

    @CFS.insertVariable()
    def __mul__(self, other):

        if isinstance(other, TensorVar) and other.step_result:
            return self.submitVariable(super().__mul__(other), [(other, other.__rmul__, (self), {})])
        return self.submitVariable(super().__mul__(other))

   

    def __rmul__(self, other):
        return self.__mul__(other)
    def __imul__(self, other):
        return self.__mul__(other)

    @CFS.insertVariable()
    def dot(self, b, out=None,fast=False):
        '''
        如果中间无需BP可以开启fast模式
        '''
        if fast:return super().dot(b)
        if isinstance(b, TensorVar) and b.step_result:
            result=self.submitVariable(np.array(super().dot(b)))
            for a_i in self:
                for b_j in b.T:
                    b_j.set_param("step_result",b.step_result).set_param("id",b.id).set_param("flow_id", b.flow_id)
                    a_i.set_param("step_result", self.step_result)
                    CFS.setLineAge(b_j, TensorVar.ldot,a_i, {}, result)
        # return self.submitVariable(np.array(super().dot(b)))
        return result
    
    def ldot(self,b):
        return self.submitVariable(np.array(b).dot(self))

    @staticmethod
    def max_depath(elem):
        depath=[]
        forEveryElemWithIndex(elem, lambda e, i, v: depath.append(len(i)))

        return max(depath)
    def _getLabel(self,label,getlabel=False):
        # sp = np.shape(self.labels)
        #
        # if len(sp)==1 and sp[0]==1 :
        #     if label==self.labels[0]:
        #         return self
        #     return None

        def f(elem, index):
            #get=lambda v,i:v if len(i)<1 else get(v[i.pop(0)],i)
            if np.array(self.labels)[index]==label:
                if getlabel:
                    return label
                return elem
            return
        return forEveryElemWithIndex(self.tolist(),f,axis=self.max_depath(self.labels),clear_null=True)

    def get(self,i):
        return np.array(super().__getitem__(i))
    def Sum(self,axis=0):
        if axis==0:
            v = [0]
            def f(e):v[0]+=e
            forEveryElem(f,self)
            return self.submitVariable([v])
        return self.submitVariable(np.array(self).sum(axis).tolist(),labels=self.labels)
    def sample(self,num=1):
        res=[]
        while len(res)<num and num<=len(self):
            r=random.randint(0, len(self) - 1)
            if r not in res:res.append(r)
        return self.submitVariable([self.toList()[r] for r in res],labels=[self.labels[r] for r in res])
 
    def get(self, item,labels=None):

        res=super().__getitem__(item)
        if not hasattr(res,"__iter__"):
            return self.submitVariable([res])
        if len(res)==1:
            return res.toList()[0]
        return res

    def __getitem__(self, item,labels=None):

        if isinstance(item,np.ndarray) or TensorVar.from_func=="creating":

            return super().__getitem__(item)
        if  isinstance(item,slice) or (isinstance(item,tuple) and slice in [type(i) for i in item]):
            litem = item
            if not isinstance(item,slice):
                litem=item[:len(np.shape(self.labels))]

            return self.submitVariable(numpy.array(self)[item].tolist(),
                                       labels=numpy.array(self.labels)[litem].tolist())

        if hasattr(item,"__iter__"):
            if len(item)==0:
                return self
            elif self.allmatch(lambda e:isinstance(e,str),item):

                return self.submitVariable(forEveryElem(self._getLabel,item),labels=forEveryElem(lambda e:self._getLabel(e,True),item))
            elif self.allmatch(lambda e:isinstance(e,int),item):

                def f(idxelems,getlabels=False):

                    elem=self.tolist().copy()
                    if getlabels:
                        elem=self.labels

                    for idxelem in idxelems:
                        if hasattr(elem,"__iter__") and not isinstance(elem,str)and len(elem)>idxelem and len(elem)!=0 :
                            elem=elem[idxelem]
                    return elem
                res=forEveryElemWithIndex(item,lambda e:f(e),axis=len(np.shape(item))-1)
                labels=forEveryElemWithIndex(item,lambda e:f(e,True),axis=len(np.shape(item))-1)
                if not isinstance(res, list):
                    res = [res]
                if not isinstance(labels, list):
                    labels = [labels]

                return self.submitVariable(res,labels=labels)

                #return super().__getitem__(item)
        elif isinstance(item,int):
            elem=super().__getitem__(item)
            if not isinstance(elem,np.ndarray):

                return elem
            if isinstance(self.labels,list) and item<len(self.labels):
                labels=self.labels[item]
            return self.submitVariable(elem,labels=labels)


    @CFS.insertVariable()
    def __truediv__(self, other):

        if isinstance(other, TensorVar)and other.step_result:
            return self.submitVariable(super().__truediv__(other),[(other,other.__rtruediv__,(self),{})])

        return self.submitVariable(super().__truediv__(other))

    def __rtruediv__(self, other):
        return self.__truediv__(self=other,other=self)

    def __itruediv__(self, other):
        return self.__truediv__(other)

    @CFS.insertVariable()
    def __neg__(self):
        return self.submitVariable(super().__neg__())


    @CFS.insertVariable()
    def __add__(self,other):
        #return self.submitVariable((np.array(self)+other).tolist())

        if isinstance(other,tuple):
            return self.submitVariable(list(self)+list(other))
        if isinstance(other, TensorVar) and other.step_result:
            return self.submitVariable(super().__add__(other), [(other, other.__radd__, (self), {})])

        return self.submitVariable(super().__add__(other))

    def __radd__(self, other):
        return self.__add__(other)

    def __iadd__(self, order):
        return self.__add__(order)

    def __mod__(self, other):
        if hasattr(other, "__iter__"):
            result=(np.array(self)%other).tolist()
            return  self.submitVariable(result)
        else:
            return self.submitVariable(forEveryElem(lambda elem:elem%other,self))


    @CFS.insertVariable()
    def __sub__(self, other):

        if isinstance(other, TensorVar) and other.step_result:
            return self.submitVariable(super().__sub__(other),[(other,other.__rsub__,(self),{})])

        return self.submitVariable(super().__sub__(other))

    # 已整合到GradSystem.caseFunc中
    # def __rsub__(self, other):
    #     return self.__add__(self=-other, other=self)

    def __isub__(self, other):
        return self.__sub__(other)


    @CFS.insertVariable()
    def __pow__(self, power, modulo=None):
        return self.submitVariable(super().__pow__(power,modulo))

    def __ipow__(self, other):
        return self.__pow__(other)

    def shuffle(self):
        r=numpy.array(list(zip(self.tolist(), self.labels)))
        numpy.random.shuffle(r)
        return self.submitVariable(r[:,0].tolist(),labels=r[:,1].tolist())

    def flat(self,tree=None):
        if tree==None:tree=self
        res = []
        for i in tree:
            if isinstance(i, list):
                res.extend(self.flat(i))
            else:
                res.append(i)
        return self.submitVariable(res)
    def groupby_label(self,axis=0,external_label=None):

        if external_label:
            labels = external_label.copy()
            self.labels=external_label
        else:
            labels = self.labels.copy()

        getv = lambda v,i: v if len(i) < 1 else getv(v[i.pop(0)],i)
        setv = lambda s,v,i:forEveryElemWithIndex(s,lambda e,j:e if j!=i else v,for_all=True)
        def f(e,i):
            if len(i)==axis:
                la=set({})
                forEveryElem(lambda e:la.add(e),getv(labels,list(i)))
                labels.clear()
                labels.extend(setv(labels,list(la),i))


                return [self[i][l]for l in la]
            return e
        #print(self.Sequece([i.tolist() for i in forEveryElemWithIndex(self,f,axis)]))


        return self.submitVariable(self.Sequece([i.tolist() for i in forEveryElemWithIndex(self,f,axis)]),labels=labels)
    
    @staticmethod
    def Sequece(vector,axis=None,full=None):
        return TensorVar.sequece(vector,axis,full)

    def sequece(self,axis=None,full=None):
        if len(self)>0:
            max_dpath=tensor().max_depath(self)
            if max_dpath>1:
                if isinstance(self,np.ndarray):
                    vectors=self.tolist()
                else:
                    vectors=list(self)
                x = [axis] if axis!=None else list(range(1,max_dpath))[::-1]
                for axis in x:
                    length=np.asarray(forEveryElemWithIndex(vectors,lambda e:len([i for i in e if i==i]),axis=axis)).ravel()
                    min_length, max_length = min(length),max(length)
                    if min_length!=max_length:
                        def f(e):
                            if full  and len(e) < max_length:
                                return list(e)+np.full((max_length-len(e)),full).tolist()
                            elif not full and len(e)> min_length:
                                return list(e[:min_length])
                            return list(e)
                        vectors=forEveryElemWithIndex(vectors,f,axis=axis)

                if type(self) == TensorVar:
                    return self.submitVariable(vectors)
                return vectors
        return self
    @staticmethod
    def full(shape,value):
        return tensor(np.full(shape,value).tolist())
    @CFS.insertVariable(d_relu)
    def relu(self):

        return  self.submitVariable((self+abs(self))/2)


    #下面形式参数中的x就是self
    @CFS.insertVariable(d_log)
    def log(x,base = np.e):
        return x.submitVariable(np.log(base)/np.log(x)) # numpy要通过换底公式求任意底数的对数

    @CFS.insertVariable(d_sigmoid)
    def sigmoid(x): 
        return  x.submitVariable(1 / (1 + np.exp(-np.array(x))))

    @CFS.insertVariable(d_softMax)
    def softMax(x):
        x_row_max = x.max(axis=-1)
        x_row_max = x_row_max.reshape(list(x.shape)[:-1] + [1])
        n = x - x_row_max
        x_exp = np.exp(n)
        x_exp_row_sum = x_exp.sum(axis=-1).reshape(list(n.shape)[:-1] + [1])
        softmax = x_exp / x_exp_row_sum
        return  x.submitVariable(softmax)

    def pad(self,padding=1):

        return self.submitVariable(numpy.pad(self,padding,mode='constant'))

    @CFS.insertVariable()
    def mean(self):
        return self.submitVariable([np.array(self).mean()],labels=[])

    def set(self,value):

        if np.shape(value)==self.shape:
            fv=np.array(value).ravel()
            [self.itemset(i, fv[i])for i in range(len(fv))]
        return self
    def toList(self):
        return self.tolist()

    def concat(self,order):
        return self.submitVariable(super().__add__(order),labels=self.labels+order.label)
    def allmatch(self, condition, item=None,real=[False]):


        if not item:
            item=self
        real[0]=True

        def wapper(elem):
            if not condition(elem):

                real[0]=False
        forEveryElem(wapper,list(item))

        return real[0]

    @CFS.insertVariable(from_func="forEveryElem")
    def forEveryElem(self,func,*args,**kwargs):
        result=forEveryElemWithIndex(self.tolist(),func,*args,**kwargs)
        return self.submitVariable(result,from_func="main")
    def toarray(self):

        return np.asarray(self)
    def apply_labels(self,key="y"):
        self.labels=self.all_labels[key]
        return self

    def set_labels(self,l,key="y"):
        self.labels =l
        self.all_labels[key]=l

    def get_labels(self,key="y"):
        return self.all_labels[key]
    def transpose(self,indexs):


        return self.submitVariable(super().transpose(indexs).tolist(),labels=self.labels)

    def reshape(self,shape):

        return self.submitVariable(super().reshape(shape),labels=self.labels)#np.reshape(self.labels,shape[:len(np.shape(self.labels))]).tolist()

    @staticmethod
    def ones( shape):
        if len(shape)==0:
            return []
        return tensor(np.ones(shape).tolist())
    def indexes(self):
        return self.submitVariable(forEveryElemWithIndex(list(self),lambda elem,index,v:list(index)))
    def track(self):
        return counter.TrackerController.track(self)

    def requireGrad(self,r):
        self.step_result=r




    @CFS.insertVariable()
    def fullGrad(self,x=None,default_func=True):
        return self.submitVariable(CFS.grad(self,x,False,default_func))

    @CFS.insertVariable()
    def Grad(self,x=None,default_func=True):
        return self.submitVariable(CFS.grad(self,x,True,default_func))

    # def shape(self):
    #     result, temp = [], self
    #     while len(temp) > 0:
    #         result.append(temp)
    #         temp = temp[0]
    #     return tuple(result)

    def save(self,path):
        pickle.dump(self,path)

    @staticmethod
    def getRandom(size,step):
        return tensor(random.randint(0,size))*step

    #拉格朗日算子
    @staticmethod
    def Lagrange(p,x=None):
        lp=list(range(len(p)))
        if not x:x=lp.copy()

        return lambda v:sum([p[j]*reduce(lambda x,y:x*y,[(v-x[i])/(x[j]-x[i])for i in lp if i!=j])for j in lp])

    @staticmethod
    def getRandomGaussian(size, sigma=1e-6, mean=0):
        '''
        正态采样
        '''
        return tensor((sigma * np.random.randn(*size) + mean),format_check=False)

    @staticmethod
    def addGradFunction(func):
        '''
        从外部增加可以直接调用的导函数
        '''
        if func !=CFS.insertVariable:
            setattr(tensor,func.__name__,CFS.insertVariable(func))
        else:
            setattr(tensor,func.__name__,func)
        return getattr(tensor,func.__name__)