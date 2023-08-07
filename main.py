
import numpy as np 
from similarity import *
from transformation import *
from copy import deepcopy
import tools
import simGen
import log
import simTree
import logging
import os
import subprocess
import sys

#TODO extract param to name img




def genetic_algo(dataset : list[list[tuple[str,str]]], nb_positives : int, db_wk_names : list[tuple[str,str]], negative_sampling : bool = False, 
                 param_algo : dict = None, plot_img_path : str = "") -> list[tuple[simGen.SimGen,list[simTree.SimTree]]] :

    param_data = {
        "similarity_functions" : similarity_functions(),
        "transformation_functions" : transformation_functions(),
        "values" : [],
        "negatives_values" : []
    }

    if not param_algo:   
        param_algo = {
            "population_size" : nb_positives,
            "nb_generation"   : 30,
            "proba_selection" : 0.4,
            "proba_crossover" : 0.2,
            "proba_mutation"  : 0.35,
            "size_regularisation" : 0.2,
            "negative_sampling_regularisation" : 0.8,
            "top_k" : 15,
            "tree_max_depth" : 10,
            "threshold" : 0.95
            
        }

    
    #TODO gerer ordre, gen dataset for each prop then gen neg, then get each neg and get each best trees for each dataset
    #maybe use helper function
    if negative_sampling:
        supposed_best_sim_functions = []
        nb_negative_example = 50
        param_algo["nb_generation"] = 5 #few generation to find an aproxima solution for the best sim function 
        for prop in dataset:
            positive_dataset = np.array(prop)[np.random.choice(np.arange(0,len(prop)),nb_positives)].tolist()
            param_data = {
                "similarity_functions" : similarity_functions(),
                "transformation_functions" : transformation_functions(),
                "values" : positive_dataset,
                "negatives_values" : []
            }
            neg_sim = simGen.SimGen(param_algo,param_data)
            neg_sim.evolve_population()
            supposed_best_sim_functions.append(get_tf_function_from_name(max(neg_sim.freq_tf[-1]))) #get most frequent sim function for the last generation

        negatives = tools.generate_NegativeSample(list(zip(*dataset)),supposed_best_sim_functions)
        negative_datasets = [list() for i in range(len(dataset))]
        for e in negatives:
            for _ in range(len(dataset)):
                negative_datasets[e[0]].append(e[1])
            
        #fix the nb of generation for our baseline param
        param_algo["nb_generation"] = 30


    results = []
    for index,prop_dataset in enumerate(dataset):
        if negative_sampling:
            param_data["negatives_values"] = negative_datasets[index] 
        param_data["values"] = prop_dataset     
        logging.debug(f"Start Genetic algo (db-wk) : {db_wk_names[index]}")
        sim = simGen.SimGen(param_algo,param_data)
        sim.evolve_population()
        tools.plot_stats_sim(sim, plot_img_path+db_wk_names[index])
        best_trees = tools.get_best_tree(sim)
        results.append((deepcopy(sim),deepcopy(best_trees)))

    return results



def __genetic_algo(dataset : list[list[tuple[str,str]]], nb_positives : int, negative_sampling : bool = False, 
                 param_algo : dict = None, plot_img_path : str = "") -> tuple[simGen.SimGen,list[simTree.SimTree], list[int]] :
    index_positives = np.random.choice(np.arange(0,len(dataset)),nb_positives)
    positive_dataset = np.array(dataset)[index_positives].tolist()

    if not(param_algo):   
        param_algo = {
            "population_size" : nb_positives,
            "nb_generation"   : 30,
            "proba_selection" : 0.4,
            "proba_crossover" : 0.2,
            "proba_mutation"  : 0.35,
            "size_regularisation" : 0.3,
            "negative_sampling_regularisation" : 0.8,
            "top_k" : 15,
            "tree_max_depth" : 10,
            "threshold" : 0.95
            
        }

    param_data = {
        "similarity_functions" : similarity_functions(),
        "transformation_functions" : transformation_functions(),
        "values" : positive_dataset,
        "negatives_values" : []
    }

    #need to generate negative sampling
    if negative_sampling:
        nb_negative_example = 50
        param_algo["nb_generation"] = 5 #few generation to find an aproxima solution for the best sim function 
        neg_sim = simGen.SimGen(param_algo,param_data)
        neg_sim.evolve_population()
        supposed_best_sim_function = get_tf_function_from_name(max(neg_sim.freq_tf[-1])) #get most frequent sim function for the last generation
        negatives = tools.generate_NegativeSample(list(map(lambda x: [x] ,dataset)),supposed_best_sim_function)
        negative_dataset = np.array(negatives)[np.random.choice(np.arange(0,len(negatives)),nb_negative_example)].tolist()
        param_data["negatives_values"] = negative_dataset      
        param_algo["nb_generation"] = 30
        

    sim = simGen.SimGen(param_algo,param_data)
    sim.evolve_population()
    tools.plot_stats_sim(sim, plot_img_path)
    best_trees = tools.get_best_tree(sim)

    return sim, best_trees, index_positives


def main():
    log.init()
    if not os.path.isdir("img"):
        subprocess.Popen(["mkdir","img"]).wait()


    key_file_path = "keys.txt"#sys.argv[1]
    data_trees = []
    train_datasets = []
    test_datasets  = []
    db_wk_names = []
    size_dataset = 1000
    size_train_dataset = 100
    with open(key_file_path,'r',encoding="UTF-8") as f:
        for line in f.read().split("\n"):
            db_prop,wk_prop = line.split(" ")
            _dataset = tools.generate_dataset(db_prop=db_prop,wk_prop=wk_prop,size=size_dataset)
            train,test = tools.train_test_datasets(_dataset, size_train=size_train_dataset)
            train_datasets.append(train)
            test_datasets.append(test)
            db_wk_names.append((db_prop,wk_prop))

    results = genetic_algo(train_datasets, nb_positives=100, negative_sampling=True, folder_path="img/"+db_prop+"/100pop_neg")

    for sim,best_trees in results:
        data_trees.append(
                {
                    "db-wk_prop_names" : (db_prop,wk_prop),
                    "dataset" : train_datasets,
                    "simTree" : sim,
                    "best_trees" : best_trees,
                }
            )



    # db_prop = "releaseDate"
    # wk_prop = "P577"
    # release_dataset = tools.generate_dataset(db_prop="releaseDate",wk_prop="P577",size=N)
    # sim_releaseDate, best_trees_releaseDate, index_positives_releaseDate = genetic_algo(release_dataset,100, True,folder_path="img/release_date/100pop")
    
    datasets = [d['dataset'] for d in data_trees ]
    names = [d["db-wk_prop_names"] for d in data_trees]
    best_trees = [d["best_trees"][0] for d in data_trees]
    #remove index_positives element from the dataset
    #need to be the same (maybe take the same seed)
    logging.debug(f"For couple {names} genetic generated tree results :")
    acc,recall = tools.test_solution(sim_trees=best_trees, dataset=datasets, threshold=0.7)



main()