def load_fitnesses(folder):
    fit_file = open(folder + "/fitnesses")
    idx_file = open(folder + "/indexes")
    
    fit_lines = fit_file.readlines()
    idx_lines = idx_lines.readlines()

    data_line = []
    iter = 0
    for fit_l,idx_l in zip(fit_lines,idx_lines):
        for fit, idx in zip(fit_l.split(","),idx_l.split(",")):
            data_line.append([iter,int(idx),int(fit)])
        iter+=1

def load_fitness_whole_exp(exp_folder):
    data_lines = []
    for variant in os.listdir(exp_folder):
        foldername = exp_folder + "/" + variant
        for replicate in os.listdir(foldername):
            fit_lines = load_fitnesses(foldername + "/" + replicate)
            data_lines += [[variant,replicate]+fit for fit in fit_lines]
    return data_lines