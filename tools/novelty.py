#! /usr/bin/python3

import random
import tools.parallel as para

def sparsness(distances,k_value=15):
    '''compute the sparsness from a list of distances'''
    sum_ = 0
    if len(distances) >= k_value:
        for dist in distances[:k_value]:
            sum_ += dist
    else:
        for dist in distances:
            sum_ += dist

    return sum_/float(k_value)


def update_archive(ind,novelty,archive,novelty_thr=0.9,adding_prob=0.4,arch_size=-1):
    '''update archive based on an novelty_thresh'''
    if arch_size == 0:
        return archive
    if novelty > novelty_thr or random.random() > adding_prob:
        archive.append(ind)
    if arch_size > 0 and len(archive) > arch_size:
        del archive[:len(archive)-arch_size]
    return archive

def distances(ind,pop,arch,dist_fct):
    '''return a lit of distances between the ind and the inds from pop and arch using dist'''
    dist = []
    for i in pop:
        dist.append(dist_fct(ind,i))
    for i in arch:
        dist.append(dist_fct(ind,i))
    dist.sort()
    return dist

def distances_parallel(pop,arch,dist_fct,cores):
    dist = para.parallel_reduce(dist_fct,para.id_split,para.list_join,pop+arch,[],cores)
    dist.sort()
    return dist