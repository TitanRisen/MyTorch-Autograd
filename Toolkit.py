import random
import numpy
import inspect

def forEveryElem(func=lambda x:x, elem=None):
    if hasattr(elem, "__iter__") and not isinstance(elem,str):
        return [forEveryElem(func, e) for e in elem]
    return func(elem)

def forEveryElemWithIndex(elem, func=lambda e,i,v:e, axis=-1,for_all=False,clear_null=False, variables=None, index=()):
    args=len(inspect.getargspec(func).args)
    if axis<0 :
        elem = list(elem)
        axis=len(numpy.shape(elem))+2+axis
    if hasattr(elem, "__iter__") and not isinstance(elem,str) and len(index) != axis:

        elem=list(elem)
        results= []
        for e in range(len(elem)):
            res=forEveryElemWithIndex(elem[e], func, axis, for_all,clear_null,variables,tuple(list(index) + [e]))

            if for_all and len(index)+1<axis: res = func(*[res, tuple(list(index) + [e]), variables][:args])
            if not clear_null or (res!=None and res!=[]):
                results.append(res)
        if for_all and len(index)==0 and axis>0:
            results=func(*[results, (), variables][:args])
            if not isinstance(results, list): results = [results]
            return results
        if len(results) == 1 and clear_null:
            return results[0]

        return results
    if for_all:
        return elem

    return func(*[elem, index, variables][:args])

class Random(object):
    uuidList=dict({})
    nm='1234567890'
    count=0
    result=[]

    @staticmethod
    def clear(group):
        Random.uuidList[group]=[]

    @staticmethod
    def uuid(samples=nm,num=2,group="default",limit=3000):
        Random.createGroup(group)
        if len(Random.uuidList[group])>=len(samples)**num/2:
            print("total times: ",Random.count)
            Random.result.append(Random.count)
            Random.count=0
            Random.clear("default")
            return False
            # raise RuntimeError("Id groups are all full.")
        count=0
        while count<limit:
            Id = "".join([random.choice(samples) for c in range(num)])
            if Id not in Random.uuidList[group]:
                print("delicate times: ", count)
                Random.count+=count
                return Random.insert(Id,group)
            count+=1
        return False

    @staticmethod
    def createGroup(group):
        if group not in Random.uuidList.keys():
            Random.uuidList[group] = []
    @staticmethod
    def insert(Id,group):
        Random.createGroup(group)
        Random.uuidList[group].append(Id)
        return Id
    @staticmethod
    def delete(Id,group):
        if Id in Random.uuidList[group]:
            Random.uuidList[group].remove(Id)