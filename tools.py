import numpy as np
import json
import os
import random
import logging
from simTree import SimTree
import matplotlib.pyplot as plt
from typing import Callable
from transformation import transformation_functions, get_tf_function_from_name
from similarity import similarity_functions, get_sf_function_from_name
from simGen import SimGen

#####################################
############ JSON methods ###########
#####################################

def json_to_dict(json_file : str, vars : tuple[str,str]) -> dict:
    with open(json_file, encoding="UTF-8") as f:
        data = json.load(f)

    json_dict = {}
    for elem in data['results']['bindings']:
        json_dict[elem[vars[0]]["value"]] = elem[vars[1]]["value"]
    
    return json_dict

def get_dict_from_json_file(json_file : str) -> dict:
    with open(json_file,'r',encoding="UTF-8") as f:
        d = json.load(f)
        return d


#####################################
########## SimTree methods ##########
##################################### 

def tree_from_list(tree_list : list) -> SimTree:
    if len(tree_list) == 3: #root
        value = tree_list[0]
        left_child = tree_list[1]
        right_child = tree_list[2]
        return SimTree(value,
                [tree_from_list(left_child),
                tree_from_list(right_child)])
    elif len(tree_list) == 2: #nodes        
        value = tree_list[0]
        child = tree_list[1]
        return SimTree(value,[tree_from_list(child)])
    else: #leaf
        value = tree_list[0]
        return SimTree(value,[])



def tree_from_str_list(tree_list : list) -> SimTree:
    if len(tree_list) == 3: #root
        value = get_sf_function_from_name(tree_list[0])
        left_child = tree_list[1]
        right_child = tree_list[2]
        return SimTree(value,
                [tree_from_str_list(left_child),
                tree_from_str_list(right_child)])
    elif len(tree_list) == 2: #nodes        
        value = get_tf_function_from_name(tree_list[0])
        child = tree_list[1]
        return SimTree(value,[tree_from_str_list(child)])
    else: #leaf
        value = tree_list[0]
        return SimTree(value,[])

def trees_from_file(tree_file : str) -> list[SimTree]:
    """From a file containing a list of tree as lists we convert and evaluate them back to objects. One line per tree

    Args:
        tree_file (str): path to file containing trees

    Returns:
        list[SimTree]: the list of trees contained in the file
    """
    trees : list[SimTree] = []
    with open(tree_file, 'r',encoding="UFT-8") as f:
        for line in f.read().split("\n"):
            trees.append(tree_from_str_list(eval(line)))
    return trees


#####################################
########## Dataset methods ##########
##################################### 

def generate_dataset(db_prop : str, wk_prop : str, size : int, folder_path : str = "../dataset_creation/data/"):
    sameAs = folder_path+wk_prop+"_"+db_prop+"/db-"+db_prop+"_wk-"+wk_prop+"_sameAs.json"
    db_support = folder_path+wk_prop+"_"+db_prop+"/dbpedia-"+db_prop+".json"
    wk_support = folder_path+wk_prop+"_"+db_prop+"/wikidata-"+wk_prop+".json"
    
    sameAs_cleaned_path = sameAs.split(".json")[0]+"_cleaned.json"
    sameAs_cleaned = {}
    if not(os.path.isfile(sameAs_cleaned_path)):
        sameAs_dict = json_to_dict(sameAs,vars=("v","e"))
        wk_support_dict = json_to_dict(wk_support,vars=("e","v"))
        for wk in wk_support_dict:
            if (db := sameAs_dict.get(wk)):
                sameAs_cleaned[db] = wk

        with open(sameAs_cleaned_path,'w',encoding="UTF-8") as f:
            json.dump(sameAs_cleaned,f)


    sameAs_dict = get_dict_from_json_file(sameAs_cleaned_path)
    #json_to_dict(sameAs,vars=("e","v"))
    db_support_dict = json_to_dict(db_support,vars=("e","v"))
    wk_support_dict = json_to_dict(wk_support,vars=("e","v"))

    result_mapping = []
    random_index = np.random.choice(np.arange(0,len(sameAs_dict),1),size,replace=False)
    index_count = 0
    for db,wk in sameAs_dict.items():
        if index_count in random_index:
            tupple = (db_support_dict.get(db),wk_support_dict.get(wk))
            result_mapping.append(tupple)
        index_count += 1
    return result_mapping

def generate_NegativeSample(dataset : list[list[tuple[str,str]]], sim_functions : list[Callable[[str,str], float]], size : int) -> list[list[tuple[str,str]]] | list[tuple[str,str]]:
    threshold = 0.7
    threshold_max_min = 0.3
    
    def compare_keys(key1 : list[tuple[str,str]],key2 : list[tuple[str,str]], sim_functions : list[Callable[[str,str], float]]) -> tuple[str,str] | None:
        sim = []
        # print(f"key1 :{key1} \t key2 : {key2}")
        for v,w,sim_function in zip(key1,key2,sim_functions):
            # print(v,w,sim_function)
            # print(f"key1:{v[0]}, key2:{w[1]}, sim {sim_function} function = {sim_function(v[0],w[1])}")
            
            sim.append(sim_function(v[0],w[1])) #v[0] is the dbpedia value, first value of tuple, w[1] is for wikidata
        if sum(map(lambda x: x>=threshold, sim)) == len(sim)-1 and (max(sim) - min(sim)) > threshold_max_min: 
            #if all but one value are above our threshold of similarity the last should not be similar because its a key and all values cannot be the same
            #and can be considered as a negative exemple
            return (np.argmax(sim),(key1[np.argmax(sim)][0], key2[np.argmax(sim)][1]))

    
    #TODO make sure we don't select the same prop every time   
    if len(dataset[0]) > 1:
        negativeSample = []
        key_selected = []
        for key in dataset: #looop over list of key properties values tuple
            for index_,key_ in enumerate(dataset):
                if key != key_ and (res := compare_keys(key,key_,sim_functions)):
                    key_selected.append((key,key_,res[1]))
                    negativeSample.append(res)
                    if len(negativeSample) == size:
                        return negativeSample, key_selected
                    # if size > len(negativeSample):
                    #     return negativeSample

    
        return negativeSample,key_selected
    
    elif len(dataset[0]) == 1:
        db_values = list(map(lambda x: list(map(lambda x: x[0],x))[0],dataset))
        wk_values = list(map(lambda x: list(map(lambda x: x[1],x))[0],dataset))
        shift = random.randint(1,len(db_values)-1) #we shift the list with itself by a random number
        index = list(np.arange(0,len(db_values)))
        shifted_index = zip(index,index[-shift:]+index[:-shift])
        res = [(db_values[i],wk_values[j]) for i,j in shifted_index]
        return res
  
def train_test_datasets(dataset : list[tuple[str,str]], size_train : int) -> tuple[list[tuple[str,str]]]:
    index_dataset = np.arange(0,len(dataset)-1)
    train_index = np.random.choice(index_dataset,size_train,replace=False)
    test_index = set(index_dataset).difference(set(train_index))
    train_dataset = np.array(dataset)[train_index].tolist()
    test_dataset = np.array(dataset)[test_index].tolist()
    return train_dataset,test_dataset
    

#####################################
########## SimGen methods ###########
##################################### 

def plot_stats_sim(sim : SimGen, save : str = "") -> None:
    _fig = plt.figure(figsize=(15,5))
    plt.subplot(1,2,1)
    pop_scores  = [np.mean(x) for x in sim.gen_scores]
    
    plt.plot(np.arange(0,len(pop_scores),1),pop_scores,marker="o",markersize=4, label="Avg sim")
    std = np.std(sim.gen_scores, axis=1)
    plt.fill_between(np.arange(0,len(pop_scores),1), pop_scores - std, pop_scores + std, alpha=0.2)

    top_k_scores  = [np.mean(x) for x in sim.gen_top_k_scores]
    plt.plot(np.arange(0,len(top_k_scores),1),top_k_scores,marker="o",markersize=4, label = "top k sim")
    std = np.std(sim.gen_top_k_scores, axis=1)
    plt.fill_between(np.arange(0,len(pop_scores),1), top_k_scores - std, top_k_scores + std, alpha=0.2)


    plt.xlabel("Generation")
    plt.ylabel("Average Similarity")
    plt.title("Similarity Evolution")
    plt.legend()
    
    plt.subplot(1,2,2)
    size_avg  = [np.mean(x) for x in sim.size_tracker]
    plt.plot(np.arange(0,len(size_avg),1),size_avg)
    plt.xlabel("Generation")
    plt.ylabel("Tree Size")
    plt.title("Tree Size Evolution")
    if save:
        plt.savefig(save+"_similarity_evolution.png")

    tf_evol = [list() for i in range(len(transformation_functions()))]
    tf_name = [f.__name__ for f in transformation_functions()]
    for freq in sim.freq_tf:
        for i,freq_val in enumerate(freq.values()):
            tf_evol[i].append(freq_val)
    _fig = plt.figure(figsize=(15,5))
    for i,tf in enumerate(tf_evol):
        evol = [s[tf_name[i]]*100 / sum(s.values()) for s in sim.freq_tf]
        plt.plot(evol,label=tf_name[i])
        #plt.title([tf_name[i]])
        #plt.show()
        plt.xlabel("Generation")
        plt.ylabel("frequency %")
        plt.title("Evolution of transformations functions distribution")
        plt.legend(bbox_to_anchor=(0.96, 0.75))
    if save:
        plt.savefig(save+"_transformation_distrib.png")

    sf_evol = [list() for i in range(len(similarity_functions()))]
    sf_name = [f.__name__ for f in similarity_functions()]
    for freq in sim.freq_sf:
        for i,freq_val in enumerate(freq.values()):
            sf_evol[i].append(freq_val)

    _fig = plt.figure(figsize=(15,5))
    for i,sf in enumerate(sf_evol):
        evol = [s[sf_name[i]]*100 / sum(s.values()) for s in sim.freq_sf]
        plt.plot(evol,label=sf_name[i])
        #plt.title([tf_name[i]])
        #plt.show()
        plt.xlabel("Generation")
        plt.ylabel("frequency %")
        plt.legend()
        plt.title("Evolution of similarities functions distribution")
    if save:
        plt.savefig(save+"_similarity_distrib.png")

def get_best_tree(sim : SimGen) -> list[SimTree]:
    (index_score := list(np.argsort(sim.population_scores))).reverse()
    print("Best sim tree score")
    for i in index_score[0:5]:
        print(sim.population_scores[i],sim.population[i])
    
    (index_sim := list(np.argsort(sim.population_similarity))).reverse()
    print("\nBest sim tree similarity")
    for i in index_sim[0:5]:
        print(sim.population_similarity[i],sim.population[i])

    return [sim.population[i] for i in index_score[0:5]]


def test_solution(sim_trees : list[SimTree], dataset : list[list[tuple[str,str]]], threshold : float) -> list[float]:

    tp = 0
    fp = 0
    tn = 0
    fn = 0
    for index1,mapping1 in enumerate(zip(*dataset)):
        db_entity = list(map(lambda x:x[0],mapping1))
        for index2,mapping2 in enumerate(zip(*dataset)):
            wk_entity = list(map(lambda x:x[1],mapping2))
            sim = []
            for tree,db,wk in zip(sim_trees,db_entity,wk_entity):
                tree.set_leafs_value([db,wk])
                sim.append(tree.compute())
            if np.mean(sim) >= threshold:
                if index1 == index2:
                    tp += 1
                else: #index1 != index2 => not a true mapping
                    fp += 1
            else:
                if index1 != index2:
                    tn += 1
                else:
                    fn += 1

    accuracy = tp / (tp+fn)
    recall   = tp / (tp+fp) if tp else 0
    logging.debug(f"accuracy : {accuracy} - recall : {recall} \n true positives : {tp} - false negatives : {fn} - true negatives : {tn} - false positives : {fp}")

    return [accuracy,recall]









# def load_trees_from_file(file : str) -> list[SimTree]:
#     trees = []
#     get_function = {
#         "jaccard_similarity" : similarity_functions()[0],
#         "cos_similarity" : similarity_functions()[1],
#         "damerau_levenshtein_dist_similarity" : similarity_functions()[2],
#         "jaro_similarity" : similarity_functions()[3],

#         "identity" : transformation_functions()[0],
#         "lowercase" : transformation_functions()[1],
#         "uppercase" : transformation_functions()[2],
#         "strip_whitespace" : transformation_functions()[3],
#         "remove_whitespace" : transformation_functions()[4],
#         "remove_punctuation" : transformation_functions()[5],
#         "flatten" : transformation_functions()[6],
#         "stem" : transformation_functions()[7],
#         "remove_stopwords" : transformation_functions()[8],
#         "tokenize" : transformation_functions()[9],
#     }
# #TODO doens't work because its a list of list etc 
#     with open(file,"r",encoding="UTF-8") as f:
#         for line in f.readlines():
#             trees.append(list(map(lambda x: get_function[x] if get_function.get(x) else x,line)))
#     return trees



# def plot_tree_stats(sim : SimGen, save = "") -> None:
#     tf_evol = [list() for i in range(len(transformation_functions()))]
#     tf_name = [f.__name__ for f in transformation_functions()]
#     for freq in sim.freq_tf:
#         for i,freq_val in enumerate(freq.values()):
#             tf_evol[i].append(freq_val)
#     _fig = plt.figure(figsize=(15,5))
#     for i,tf in enumerate(tf_evol):
#         evol = [s[tf_name[i]]*100 / sum(s.values()) for s in sim.freq_tf]
#         plt.plot(evol,label=tf_name[i])
#         #plt.title([tf_name[i]])
#         #plt.show()
#         plt.xlabel("Generation")
#         plt.ylabel("frequency %")
#         plt.title("Evolution of transformations functions distribution")
#         plt.legend(bbox_to_anchor=(0.96, 0.75))
#     if save:
#         plt.savefig(save+"_tf.png")

#     sf_evol = [list() for i in range(len(similarity_functions()))]
#     sf_name = [f.__name__ for f in similarity_functions()]
#     for freq in sim.freq_sf:
#         for i,freq_val in enumerate(freq.values()):
#             sf_evol[i].append(freq_val)

#     _fig = plt.figure(figsize=(15,5))
#     for i,sf in enumerate(sf_evol):
#         evol = [s[sf_name[i]]*100 / sum(s.values()) for s in sim.freq_sf]
#         plt.plot(evol,label=sf_name[i])
#         #plt.title([tf_name[i]])
#         #plt.show()
#         plt.xlabel("Generation")
#         plt.ylabel("frequency %")
#         plt.legend()
#         plt.title("Evolution of similarities functions distribution")
#     if save:
#         plt.savefig(save+"_sf.png")
    