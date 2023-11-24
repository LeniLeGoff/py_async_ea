import os
import numpy as np

def load_fitness_whole_exp(exp_folder):
    data_lines = []
    for variant in os.listdir(exp_folder):
        foldername = exp_folder + "/" + variant
        for replicate in os.listdir(foldername):
            fit_lines = load_fitnesses(foldername + "/" + replicate)
            data_lines += [[variant,replicate]+fit for fit in fit_lines]
    return data_lines


def load_fitnesses(folder):
    fit_file = open(folder + "/fitnesses")
    idx_file = open(folder + "/parent_indexes")
    
    fit_lines = fit_file.readlines()
    idx_lines = idx_file.readlines()

    fit_file.close()
    idx_file.close()

    data_line = []
    iter = 0
    prev_idx = []
    for fit_l,idx_l in zip(fit_lines,idx_lines):
        for fit, idx in zip(fit_l.split(","),idx_l.split(",")):
            data_line.append([iter,int(idx),float(fit)])
        iter+=len(list(set(idx_l.split(',')) ^ set(prev_idx)))
        prev_idx = idx_l.split(",")
    return data_line

def cap_to(N,n):
    return round(n/N)

def load_fitnesses_agg(folder):
    fit_file = open(folder + "/fitnesses")
    idx_file = open(folder + "/parent_indexes")
    new_idx_file = open(folder + "/new_ind_indexes")
    time_file = open(folder + "/time_data")
    
    fit_lines = fit_file.readlines()
    idx_lines = idx_file.readlines()
    new_idx_lines = new_idx_file.readlines()
    time_lines = time_file.readlines()
    init_time = float(time_lines[0])
    time_data = [float(time_lines[i+1]) - float(time_lines[i]) for i in range(len(time_lines[:-1]))]
    cum_time_data = [float(t) - init_time for t in time_lines]
    disc_time_data = [10000*round(t/10000) for t in cum_time_data]

    fit_file.close()
    idx_file.close()

    data_line = []
    iter = 0
    prev_iter = 0
    prev_idx = []
    new_ind = 0
    for fit_l,idx_l, n_idx_l, t,ct,dt in zip(fit_lines,idx_lines, new_idx_lines, time_data,cum_time_data,disc_time_data):
        fitnesses = [float(fit) for fit in fit_l.split(",")]
        if(prev_iter == cap_to(200,iter)):
            new_ind += len(list(set(idx_l.split(',')) - set(prev_idx)))
        else:
            new_ind = len(list(set(idx_l.split(',')) - set(prev_idx)))
        data_line.append([t,ct,dt,200*cap_to(200,iter),iter,new_ind,np.mean(fitnesses),np.median(fitnesses),np.std(fitnesses),np.max(fitnesses),np.min(fitnesses)])
        prev_iter = cap_to(200,iter)
        iter+=len(n_idx_l.split(','))
        prev_idx = idx_l.split(",")
    return data_line

def load_learning_data_agg(folder):
    lt_file = open(folder + "/learning_trials")
    ld_file = open(folder + "/learning_delta")
    idx_file = open(folder + "/parent_indexes")
    time_file = open(folder + "/time_data")

    lt_lines = lt_file.readlines()
    ld_lines = ld_file.readlines()
    idx_lines = idx_file.readlines()
    time_lines = time_file.readlines()
    init_time = float(time_lines[0])
    time_data = [float(time_lines[i+1]) - float(time_lines[i]) for i in range(len(time_lines[:-1]))]
    cum_time_data = [float(t) - init_time for t in time_lines]
    disc_time_data = [10000*round(t/10000) for t in cum_time_data]

    lt_file.close()
    ld_file.close()
    idx_file.close()

    data_line = []
    iter = 0
    prev_idx = []
    for lt_l, ld_l,idx_l, t, ct, dc in zip(lt_lines,ld_lines,idx_lines,time_data,cum_time_data, disc_time_data):
        deltas = [float(ld) for ld in ld_l.split(",")]
        trials = [int(lt) for lt in lt_l.split(",")]
        data_line.append([t,ct,dc,200*cap_to(200,iter),iter, \
                        np.mean(deltas),np.median(deltas),np.var(deltas),np.max(deltas),np.min(deltas), \
                        np.mean(trials),np.median(trials),np.var(trials),np.max(trials),np.min(trials),])
        iter+=len(list(set(idx_l.split(',')) - set(prev_idx)))
        prev_idx = idx_l.split(",")
    return data_line

def load_morph_norms_agg(folder):
    mn_file = open(folder + "/morph_norms")
    idx_file = open(folder + "/parent_indexes")
    time_file = open(folder + "/time_data")
    
    mn_lines = mn_file.readlines()
    idx_lines = idx_file.readlines()
    time_lines = time_file.readlines()
    init_time = float(time_lines[0])
    time_data = [float(time_lines[i+1]) - float(time_lines[i]) for i in range(len(time_lines[:-1]))]
    cum_time_data = [float(t) - init_time for t in time_lines]
    disc_time_data = [10000*round(t/10000) for t in cum_time_data]

    mn_file.close()
    idx_file.close()

    data_line = []
    iter = 0
    prev_iter = 0
    prev_idx = []
    new_ind = 0
    for mn_l,idx_l, t,ct,dt in zip(mn_lines,idx_lines, time_data,cum_time_data,disc_time_data):
        norms = [float(mn) for mn in mn_l.split(",")]
        if(prev_iter == cap_to(200,iter)):
            new_ind += len(list(set(idx_l.split(',')) - set(prev_idx)))
        else:
            new_ind = len(list(set(idx_l.split(',')) - set(prev_idx)))
        data_line.append([t,ct,dt,200*cap_to(200,iter),new_ind,np.mean(norms),np.median(norms),np.var(norms),np.max(norms),np.min(norms)])
        prev_iter = cap_to(200,iter)
        iter+=len(list(set(idx_l.split(',')) - set(prev_idx)))
        prev_idx = idx_l.split(",")
    return data_line

def load_controller_logs(folder):
    for dir in os.listdir(folder):
        if dir.split("_")[0] != "controller":
            continue
        for file in os.listdir(folder + "/" + dir):
            if file.split("_")[1] != "log":
                continue
            log_file = open(file)
            log_lines = log_file.readlines()
            if len(log_lines) == 0:
                pass
            else:
                fitnesses = [float(f) for f in log_lines.split(',')[-1]]
                np.var(fit[len(fit)/2:])
                
