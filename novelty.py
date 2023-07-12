#! /usr/bin/python3

import random

def sparsness(distances,k_value=15):
    '''compute the sparsness from a list of distances'''
    sum = 0
    if len(distances) >= k_value:
        for dist in distances[:k_value]:
            sum += dist
    else:
        for dist in distances:
            sum += dist

    return dist/float(k_value)


def update_archive(ind,novelty,archive,novelty_thr=0.9,adding_prob=0.4,arch_size=-1):
    '''update archive based on an novelty_thresh'''
    if arch_size == 0:
        return archive
    if novelty > novelty_thr or random.random() > adding_prob:
        archive.append(ind)
    if arch_size > 0 and len(archive) > arch_size:
        archive = archive[:len(archive)-arch_size]
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