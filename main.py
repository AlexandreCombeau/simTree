
import numpy as np 
from similarity import *
from transformation import *
from copy import deepcopy
import tools
import simGen
import simTree
import log
from simTree import SimTree
import logging
import os
import subprocess
import json

NB_generation = 20

def genetic_algo(dataset : list[list[tuple[str,str]]], nb_positives : int, prop_names : list[tuple[str,str]], negative_sampling : bool = False, 
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
            "nb_generation"   : NB_generation,
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
            supposed_best_sim_functions.append(get_sf_function_from_name(max(neg_sim.freq_sf[-1]))) #get most frequent sim function for the last generation

        negatives = tools.generate_NegativeSample(list(zip(*dataset)),supposed_best_sim_functions,size=50)
        negative_datasets = [list() for i in range(len(dataset))]
        for e in negatives:
            for _ in range(len(dataset)):
                #for multi prop keys, e[0] is the prop index in the key
                if len(dataset) > 1:
                    negative_datasets[e[0]].append(e[1])
                else:
                    negative_datasets = [negatives]
            
        #fix the nb of generation for our baseline param
        param_algo["nb_generation"] = NB_generation


    results = []
    for index,prop_dataset in enumerate(dataset):
        if negative_sampling:
            param_data["negatives_values"] = negative_datasets[index] 
        param_data["values"] = prop_dataset     
        logging.info(f"Start Genetic algo (db-wk) : {prop_names[index]}")
        sim = simGen.SimGen(param_algo,param_data)
        sim.evolve_population()
        tools.plot_stats_sim(sim, plot_img_path+prop_names[index][0])
        best_trees = tools.get_best_tree(sim)
        results.append((deepcopy(sim),deepcopy(best_trees)))

    return results


def run_simGen(key_file_path : str):
    log.init()
    if not os.path.isdir("img"):
        subprocess.Popen(["mkdir","img"]).wait()


    key_file_path = key_file_path #"keys.txt"#sys.argv[1]
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

    results = genetic_algo(train_datasets, 100,db_wk_names ,negative_sampling=True, plot_img_path = "img/")

    for sim,best_trees in results:
        data_trees.append(
                {
                    "db-wk_prop_names" : (db_prop,wk_prop),
                    "best_trees" : best_trees,
                }
            )

    # db_prop = "releaseDate"
    # wk_prop = "P577"
    # release_dataset = tools.generate_dataset(db_prop="releaseDate",wk_prop="P577",size=N)
    # sim_releaseDate, best_trees_releaseDate, index_positives_releaseDate = genetic_algo(release_dataset,100, True,folder_path="img/release_date/100pop")
    names = [d["db-wk_prop_names"] for d in data_trees]
    best_trees = [d["best_trees"][0] for d in data_trees]
    logging.info(f"For couple {names} genetic generated tree results :")
    acc,recall = tools.test_solution(sim_trees=best_trees, dataset=test_datasets, threshold=0.7)
    results = {}
    _sim_trees ={}
    for d,name in zip(data_trees,names):
        _sim_trees[str(name)] = str(d["best_trees"])
    results["best_trees"] = _sim_trees
    results["accuracy"] = acc
    results["recall"] = recall
    #TODO need to calc f-measure
    #TODO to test
    with open("results/key_name.json", "w", encoding="UTF-8") as f:
        json.dump(results, f)

def run_simGen_Vickey(key : list[tuple[str,str]], datasets: list[list[tuple[str,str]]]) -> dict[str,str]:
    
    if not os.path.isdir("img"):
        subprocess.Popen(["mkdir","img"]).wait()
    results = genetic_algo(datasets, 100, key, negative_sampling=True, plot_img_path = "img/")
    
    dict_prop_tree = {} 
    for res,prop in zip(results,key): 
        _sim,best_trees = res
        dict_prop_tree[str(prop)] = best_trees[0]

    return dict_prop_tree


def main():
    log.init()
    logging.info(f"Start main")
    #not for now, just use key list given
        # #return a list of key name
        # def sackey(dataset1_path : str, dataset2_path : str) -> list[list[str]]: 
        #     pass
        # #run sackey 
        # key_list = sackey()

    CLASSE = "actor"
    DATASET1_graph = "http://dbpedia/graph"
    DATASET2_graph = "http://yago/graph"
 

    key_file_path_dataset1 = "../datasets/Actor/Actor/DB_Actor.keys"
    key_file_path_dataset2 = "../datasets/Actor/Actor/YAGO_Actor.keys"
    # output_data_folder_path = sys.argv[3]
    root_folder_path = "../datasets/Actor/Actor/" 
    goldstandard_file_path = "../datasets/Actor/Actor/Actor.Goldstandard.txt" 

    logging.info(f"Start prop filtering")
    #compute list of valid keys which are not empty and have a mapping
    logging.info(f"Start mapped key")
    

    if os.path.isfile("data/valid_keys_list.json"):
        with open("data/valid_keys_list.json") as json_file:
            valid_keys_list = json.load(json_file)
    else:
        mapped_key_list = tools.mapped_key(key_file_path_dataset1,key_file_path_dataset2)
        logging.info(f"Start not empty keys")
        valid_keys_list = tools.not_empty_keys(mapped_key_list, sparql_graph=DATASET1_graph)
        logging.info(f"valid key list : {valid_keys_list}")

        with open("data/valid_keys_list.json","w",encoding="UTF-8") as f:
            json.dump(valid_keys_list,f)
    
    
    #for all these keys compute best tree
    unique_key_list = []
    for key in valid_keys_list:
        for prop in key:
            if prop not in unique_key_list:
                unique_key_list.append(prop)
    
    logging.info(f"Unique key list find : {unique_key_list}")
    #l'approche key par key ou prop par prop est définie avec le for et l'attribution dans le dict

    dict_prop_trees : dict[str,SimTree]= {} 

    flag_all_tree_computed = False
    if os.path.isfile("data/simTree_"+CLASSE+".json"):
        with open("data/simTree_"+CLASSE+".json") as json_file:
            dict_prop_trees = json.load(json_file)
            if all([prop in dict_prop_trees for prop in unique_key_list]):
                flag_all_tree_computed = True
            for prop in dict_prop_trees:
                dict_prop_trees[prop] = tools.tree_from_str_list(eval(dict_prop_trees[prop]))
    #flag to test if all tree were already computed then avoid doing it again
    if not flag_all_tree_computed:
        #we have a best tree for each tuple 
        logging.info(f"Start simGen for all keys")
        for key in valid_keys_list:
            if not all([prop in dict_prop_trees for prop in key]):
                logging.info(f"Start generate dataset for key : {key}")
                dataset1_key = key
                dataset2_key = key
                ouput_file_path_d1 = root_folder_path+str(dataset1_key)
                # logging.info(f"Start generate dataset 1:")
                tools.generate_dataset_vickey(dataset1_key, output_path=ouput_file_path_d1,sparql_graph=DATASET1_graph, limit= 100)
                ouput_file_path_d2 = root_folder_path+str(dataset2_key)
                # logging.info(f"Start generate dataset 2:")
                tools.generate_dataset_vickey(dataset2_key, output_path=ouput_file_path_d2,sparql_graph=DATASET2_graph, limit= 100)
                
                datasets = tools.get_dataset_from_json(ouput_file_path_d1,ouput_file_path_d2, goldstandard_file_path)
                
                if not os.path.isfile(ouput_file_path_d1+".json"):
                    subprocess.Popen(["del",ouput_file_path_d1+".json"]).wait()
                if not os.path.isfile(ouput_file_path_d2+".json"):
                    subprocess.Popen(["del",ouput_file_path_d2+".json"]).wait()

                logging.info(f"Start simGen key : {key}")
                dict_prop_best_trees = run_simGen_Vickey(key,datasets=datasets)

                for prop in key:
                    logging.info(f" Prop : {prop} - best tree : {dict_prop_best_trees[str(prop)]} ")
                    dict_prop_trees[str(prop)] =  dict_prop_best_trees[str(prop)]    
                with open("data/simTree_"+CLASSE+".json",'w',encoding="UTF-8") as json_file:
                    for prop in dict_prop_trees:
                        dict_prop_trees[prop] = str(dict_prop_trees[prop])
                    json.dump(dict_prop_trees,json_file)      
    
    if os.path.isfile("data/simTree_"+CLASSE+".json"):
        with open("data/simTree_"+CLASSE+".json") as json_file:
            dict_prop_trees = json.load(json_file)
            for prop in dict_prop_trees:
                dict_prop_trees[prop] = tools.tree_from_str_list(eval(dict_prop_trees[prop]))

# We save all data in json file to remove all graphdb interaction 
# and port the computation into a more powerfull server
    if os.path.isfile("data/E1.json"):
        with open("data/E1.json") as json_file:
            E1 = json.load(json_file)
    else:
        E1 = tools.get_entities_prop(sparql_graph=DATASET1_graph, goldstandard_file_path=goldstandard_file_path)
        with open("data/E1.json","w",encoding="UTF-8") as f:
            json.dump(E1,f)

    if os.path.isfile("data/E2.json"):
        with open("data/E2.json") as json_file:
            E2 = json.load(json_file)
    else:
        E2 = tools.get_entities_prop(sparql_graph=DATASET2_graph,goldstandard_file_path=goldstandard_file_path)
        with open("data/E2.json","w",encoding="UTF-8") as f:
            json.dump(E2,f)


    if os.path.isfile("data/e1_full_dataset.json"):
        with open("data/e1_full_dataset.json") as json_file:
            e1_full_dataset = json.load(json_file)
    else:
        e1_full_dataset = tools.full_dataset(DATASET1_graph)
        with open("data/e1_full_dataset.json","w",encoding="UTF-8") as f:
            json.dump(e1_full_dataset,f)

    if os.path.isfile("data/e2_full_dataset.json"):
        with open("data/e2_full_dataset.json") as json_file:
            e2_full_dataset = json.load(json_file)
    else:
        e2_full_dataset = tools.full_dataset(DATASET2_graph)
        with open("data/e2_full_dataset.json","w",encoding="UTF-8") as f:
            json.dump(e2_full_dataset,f)

    
    logging.info(f"Size DBpedia : {len(E1)}, Size Yago : {len(E2)}")        

    logging.info(f"Start test loop")
    similarity_results_dict = {}
    if os.path.isfile("data/similarity_dict_"+CLASSE+".json"):
        with open("data/similarity_dict_"+CLASSE+".json") as json_file:
            similarity_results_dict = json.load(json_file)
    else:
        for i,e1 in enumerate(E1):       
            
            # E2_k = tools.filter_dataset(E2,keys_supported_e1)
            logging.info(f"start : {e1} with : {len(E2)}. Item {i} / {len(E1)}")
            c = 0
            for e2 in E2:
                usable_keys = []
                keys_supported_e1 = list(map(lambda x:not set(x).difference(set(E1[e1])), valid_keys_list))
                keys_supported_e2 = list(map(lambda x:not set(x).difference(set(E2[e2])), valid_keys_list))
                for key, k1,k2 in zip(valid_keys_list,keys_supported_e1,keys_supported_e2):
                    if k1 and k2:
                        usable_keys.append(key)
                if usable_keys == []:#is no key found we will compare all possibles properties
                    usable_keys = list(set(E1[e1]).intersection(set(E2[e2]))) 
                    usable_keys = [[prop] for prop in usable_keys] #convert single prop to 2D list to match key structure
                if usable_keys == []:#no common properties => we assume their similarity is 0
                    similarity_results_dict[e1] = {}
                    similarity_results_dict[e1][e2] = {}
                    similarity_results_dict[e1][e2]["empty"] = 0
                else:
                    for key in usable_keys:
                        # logging.info(usable_keys)
                        if not e1 in similarity_results_dict:
                            similarity_results_dict[e1] = {}
                        if not similarity_results_dict[e1].get(e2):
                            similarity_results_dict[e1][e2] = {}
                        similarity_results_dict[e1][e2][str(key)] = tools.compute_similarity(e1_full_dataset[e1],e2_full_dataset[e2],key,dict_prop_trees,[DATASET1_graph,DATASET2_graph]) 
                c+=1
            logging.info(F"{e1} -> {c} == {len(E2)}")
        
        #save huge dict into json
        with open("data/similarity_dict_"+CLASSE+".json","w",encoding="UTF-8") as f:
            json.dump(similarity_results_dict,f)

    logging.info(f"Start stats loop")    
    mapped_entities = []    
    for i,e1 in enumerate(similarity_results_dict):
        logging.info(f"start : {e1} . Item {i} / {len(similarity_results_dict)}")
        best_key = ""
        value_best_key = 0
        best_entity = ""
        value_best_entity = 0
        for e2 in similarity_results_dict[e1]: #iter over all e2 mapped with e1
            for key,value in similarity_results_dict[e1][e2].items(): #iter over all (key,value sim) of e2
                if value > value_best_key:
                    value_best_key = value
                    best_key = key
            if value_best_key > value_best_entity:
                value_best_entity = value_best_key
                best_entity = e2  
        mapped_entities.append( (e1,best_entity,best_key,value_best_key) )

    logging.info("Acc calcul")
    goldstandard_entity_mapping = {}
    with open(goldstandard_file_path,"r", encoding="UTF-8") as f:
        for line in f.read().split("\n"):
            if line:
                e1,e2 = line.split("\t")
                goldstandard_entity_mapping[e1] = e2 
    tp = 0
    fn = 0
    fp = 0
    tn = 0
    threshold = 0.85 #TODO need to define threshold
    for e1,e2,_,v in mapped_entities:
        if e1 in goldstandard_entity_mapping:
            if e2 == goldstandard_entity_mapping[e1]:
                if v >= threshold:
                    tp += 1
                else:
                    fn += 1
            else:
                if v>= threshold:
                    fp += 1
                else:
                    tn += 1

    logging.info(f"true positives : {tp} - false negatives : {fn} - true negatives : {tn} - false positives : {fp}")

    if tp == 0 or fn == 0:
        precision = 0
    else:
        precision = tp / (tp+fn) 

    if tp+fp == 0:
        recall = 0
    else:
        recall   = tp / (tp+fp) 

    if precision == 0 or recall == 0:
        f_measure = 0
    else:
        f_measure = 2* (( precision * recall)/(precision + recall)) 

    logging.info(f"f-measure : {f_measure} - precision : {precision} - recall : {recall} \n true positives : {tp} - false negatives : {fn} - true negatives : {tn} - false positives : {fp}")

    with open("results/"+CLASSE+"_results.txt","w") as f:
        f.write(f"f-measure : {f_measure} - precision : {precision} - recall : {recall} \n true positives : {tp} - false negatives : {fn} - true negatives : {tn} - false positives : {fp}")
    return precision,recall,f_measure,tp,fn,fp,tn

    #generation des arbres
    #soit on fait prop par prop donc clef unique
    #soit on fait key par key avec un dict props : best tree 