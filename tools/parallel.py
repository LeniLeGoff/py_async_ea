#! /usr/bin/python3

import multiprocessing as mp

def parallel_for(fct,set_,cores=8):
    with mp.Pool(processes=8) as pool:
        return pool.map(fct,set_)

class ParallelReduce:
    def __init__(self,fct,split,join,set_,res,cores=8):
        self.fct = fct
        self.join = join
        self.set = set_
        self.res = res
        self.split = split

    def callback(self,results):
        self.res = self.join(self.res,results)

    def apply(self,cores):
        pool = mp.Pool(processes=cores)
        for elt in self.split(self.set):
            pool.apply_async(self.fct,elt,callback=self.callback)
        pool.close()
        pool.join()

    def apply_ordered(self,cores):
        pool = mp.Pool(processes=cores)
        for elt in self.split(self.set):
            worker = pool.apply_async(self.fct,elt,callback=self.callback)
            worker.get()
        pool.close()
        pool.join()        


def parallel_reduce(fct,split,join,set_,res,cores=8):
    pr = ParallelReduce(fct,split,join,set_,res)
    pr.apply(cores)
    return pr.res
    
def parallel_reduce_ordered(fct,split,join,set_,res,cores=8):
    pr = ParallelReduce(fct,split,join,set_,res)
    pr.apply_ordered(cores)
    return pr.res

def id_split(a):
    return [(elt,) for elt in a]

def list_join(l,elt):
    l.append(elt)
    return l

#---tests---
def f(x):
    return x*x
def test_parallel_for():
    a = [i for i in range(20)]
    a = parallel_for(f,a)
    print(a)

def sum_(x,y):
    return x + y

def sum_split(set_):
    return [(set_[i], set_[i+1]) for i in range(0,len(set_),2)]

def test_parallel_reduce():
    print(parallel_reduce(sum_,sum_split,sum_,range(1000),0))
    print(parallel_reduce_ordered(sum_,sum_split,list_join,range(1000),[]))

if __name__ == '__main__':
    print("test parallel_for")
    test_parallel_for()
    print("test prallel_reduce")
    test_parallel_reduce()