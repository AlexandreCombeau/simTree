import numpy as np
import json
import os
import pathlib
import random
import logging
from simTree import SimTree
import matplotlib.pyplot as plt
from typing import Callable
from transformation import transformation_functions, get_tf_function_from_name
from similarity import similarity_functions, get_sf_function_from_name
from simGen import SimGen
from SPARQLWrapper import SPARQLWrapper, JSON
from copy import deepcopy

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
        value = tree_list
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
    logging.info("start neg sampling")
    
    def compare_keys(key1 : list[tuple[str,str]],key2 : list[tuple[str,str]], sim_functions : list[Callable[[str,str], float]]):
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

                        return negativeSample
                    # if size > len(negativeSample):
                    #     return negativeSample
        return negativeSample
    
    elif len(dataset[0]) == 1:
        db_values = list(map(lambda x: list(map(lambda x: x[0],x))[0],dataset))
        wk_values = list(map(lambda x: list(map(lambda x: x[1],x))[0],dataset))
        shift = random.randint(1,len(db_values)-1) #we shift the list with itself by a random number
        index = list(np.arange(0,len(db_values)))
        shifted_index = zip(index,index[-shift:]+index[:-shift])
        res = [(db_values[i],wk_values[j]) for i,j in shifted_index]
        return res
  
def train_test_datasets(dataset : list[tuple[str,str]], size_train : int) -> tuple[list[tuple[str,str]], list[tuple[str,str]]]:
    index_dataset = np.arange(0,len(dataset)-1)
    train_index = np.random.choice(index_dataset,size_train,replace=False)
    test_index = set(index_dataset).difference(set(train_index))
    train_dataset = np.array(dataset)[list(train_index)].tolist()
    test_dataset = np.array(dataset)[list(test_index)].tolist()
    return train_dataset,test_dataset
    

#####################################
########## SimGen methods ###########
##################################### 

def plot_stats_sim(sim : SimGen, save : str = "") -> None:

    #Plot scoring evolution
    pop_score  = [np.mean(x) for x in sim.gen_scores]
    
    plt.plot(np.arange(0,len(pop_score),1),pop_score,marker="o",markersize=4, label="Avg score")
    std = np.std(sim.gen_scores, axis=1)
    plt.fill_between(np.arange(0,len(pop_score),1), pop_score - std, pop_score + std, alpha=0.2)

    top_k_score  = [np.mean(x) for x in sim.gen_top_k_score]
    plt.plot(np.arange(0,len(top_k_score),1),top_k_score,marker="o",markersize=4, label = "top k score")
    std = np.std(sim.gen_top_k_score, axis=1)
    plt.fill_between(np.arange(0,len(pop_score),1), top_k_score - std, top_k_score + std, alpha=0.2)


    plt.xlabel("Generation")
    plt.ylabel("Average Score")
    plt.title("Score Evolution")
    plt.legend()
    
    if save:
        plt.savefig(save+"_score_evolution.png")

    #Plot similarity evolution 
    _fig = plt.figure(figsize=(15,5))
    plt.subplot(1,2,1)
    pop_sim  = [np.mean(x) for x in sim.gen_sim]
    
    plt.plot(np.arange(0,len(pop_sim),1),pop_sim,marker="o",markersize=4, label="Avg sim")
    std = np.std(sim.gen_sim, axis=1)
    plt.fill_between(np.arange(0,len(pop_sim),1), pop_sim - std, pop_sim + std, alpha=0.2)

    top_k_sim  = [np.mean(x) for x in sim.gen_top_k_sim]
    plt.plot(np.arange(0,len(top_k_sim),1),top_k_sim,marker="o",markersize=4, label = "top k sim")
    std = np.std(sim.gen_top_k_sim, axis=1)
    plt.fill_between(np.arange(0,len(pop_sim),1), top_k_sim - std, top_k_sim + std, alpha=0.2)

    

    plt.xlabel("Generation")
    plt.ylabel("Average Similarity")
    plt.title("Similarity Evolution")
    plt.legend()
    
    #Plot size evolution
    plt.subplot(1,2,2)
    size_avg  = [np.mean(x) for x in sim.size_tracker]
    plt.plot(np.arange(0,len(size_avg),1),size_avg)
    plt.xlabel("Generation")
    plt.ylabel("Tree Size")
    plt.title("Tree Size Evolution")
    if save:
        plt.savefig(save+"_similarity_evolution.png")

    #Plot frequency of similarity and transformation functions
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


#####################################
########### Test methods ############
##################################### 

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
    logging.info(f"accuracy : {accuracy} - recall : {recall} \n true positives : {tp} - false negatives : {fn} - true negatives : {tn} - false positives : {fp}")

    return [accuracy,recall]


#####################################
#### Test Vikey dataset methods #####
##################################### 

#TODO
#modif pour chaque classe
def propName_to_propUri(prop_name : str) -> str:
    """From a property name return the uri associated

    Args:
        prop_name (str) : name of the property with prefix

    Returns:
        str: uri 
    """
    if "skos" in prop_name:
        return "<http://www.w3.org/2004/02/skos/core#preflabel>"
    else:
        return "<http://simGen/actor/"+prop_name+">"

def name_from_uri(uri : str) -> str:
    """From a uri return the name

    Args:
        uri (str): _description_

    Returns:
        str: _description_
    """
    if "skos" in uri:
        return "skos:"+uri.split("#")[-1]
    else:
        return uri.split("/")[-1]

def query_maker(props : list[str], limit : int, sparql_graph :str, count : bool = False) -> str:
    """ Generate a sparql query to get all entity for a list of properties

    Args:
        props (list[str]): list of properties, [p1,p2,...]
        limit (int): size of the dataset
        sparql_graph (str): name of the sparql graph we want to query_
        count (bool, optional): If we only want to see the count to check if a property has a support. Defaults to False.

    Returns:
        str: return the sparql query in string format
    """
    prop_counter = 1
    
    select_var_str = ""
    bind_str = ""
    where_str = ""
    for p in props:
        
        bind_str += "bind("+p+" as ?p"+str(prop_counter)+"). "
        where_str += "?e ?p"+str(prop_counter)+" ?v"+str(prop_counter)+". "
        select_var_str +=" ?p"+str(prop_counter)+" ?v"+str(prop_counter)
        prop_counter += 1 

    query = "select distinct ?e "+select_var_str+" where { graph <"+sparql_graph+"> { "+bind_str+where_str+"} } limit "+str(limit)
    if count:
        query = "select distinct (count(?e) as ?c) where { "+bind_str+where_str+" } limit "+str(limit)
    return query 

def sparql_query(query : str, endpoint : str ="http://Jools:7200/repositories/wod" ) -> dict:
    """Query the sparql database

    Args:
        query (str): _description_
        endpoint (_type_, optional): sparql endpoint local adress. Defaults to "http://Jools:7200/repositories/wod".

    Raises:
        Exception: _description_

    Returns:
        dict: result dictionary 
    """
    sparql = SPARQLWrapper(
        endpoint
    )
    sparql.setReturnFormat(JSON)
    sparql.setQuery(query)

    try:
        ret = sparql.queryAndConvert()
        return ret
    except Exception as e:
        print(e)
        raise Exception("Error in query")

def generate_dataset_vickey(props : list[str], output_path : str,sparql_graph : str, limit : int = 1000) -> None:
    """Generate a json file from a list a property

    Args:
        props (list[str]): list of property we want to query
        output_path (str): path of the output json file 
        sparql_graph (str): _description_
        limit (int, optional): _description_. Defaults to 1000.
    """
    props_uri = [propName_to_propUri(p) for p in props]
    query = query_maker(props_uri, limit=limit, sparql_graph=sparql_graph)
    logging.info(f"Query for prop : {props}, sparql : {query}")
    res = sparql_query(query)
    entities = {}
    entity_dict = {}

    current_entity = ""
    for entity in res["results"]["bindings"]:
        for var in res["head"]["vars"]:
            v =  entity[var]["value"]
            # print(entity_dict)
            
            # print(v)
            if var == "e":
                if not entities.get(name_from_uri(v)):
                    entities[name_from_uri(v)] = {
                        "name" : name_from_uri(v),
                        "uri"  : v,
                        "prop" : []
                    }
                current_entity = name_from_uri(v)

                # entity_dict["name"] = name_from_uri(v)
                # entity_dict["uri"] = v
                # entity_dict["prop"] = []
            elif "p" in var:
                nb_prop = var[-1] #each prop is of the form ?pN, N being a counter, ?v2 is the value for the prop ?p2
                #add if this prop dict is not init, if his uri is not in the list of added prop
                # if v not in [dict_prop["uri"] for dict_prop in entity_dict["prop"]]:
                if v not in [dict_prop["uri"] for dict_prop in entities[current_entity]["prop"]]:
                # if not entity_dict.get("prop"): #doens't exist, need to init the dict
                    entities[current_entity]["prop"].append( {
                        "name" : name_from_uri(v),
                        "uri" : v,
                        "value" : []
                    })
            elif "v" in var:
                nb_prop = var[-1]
                #entity_dict["prop"][nb_prop-1] give us the correct list elem corresponding to the prop number
                #print(len(entity_dict["prop"]), var, entity_dict)
                if entity["v"+str(nb_prop)]["value"] not in entities[current_entity]["prop"][int(nb_prop)-1]["value"]:
                    entities[current_entity]["prop"][int(nb_prop)-1]["value"].append(entity["v"+str(nb_prop)]["value"])
        # entities[name_from_uri(entity["e"]["value"])] = deepcopy(entity_dict)

    with open(output_path+".json","w",encoding="UTF-8") as f:
        json.dump(entities,f, indent=4, separators=(',',':'))

def get_dataset_from_json(dataset1_file_path : str, dataset2_file_path : str, goldstandard_file_path : str) -> list[list[tuple[str,str]]]:
    """Parse a json file to generate a dataset for mapped entities

    Args:
        dataset1_file_path (str): _description_
        dataset2_file_path (str): _description_
        goldstandard_file_path (str): _description_

    Returns:
        list[list[tuple[str,str]]]: _description_
    """
    def parse_json_dataset_to_dict(dataset_file_path :str) -> dict[str,dict[str,str]]:
        dict_propTo_nameTo_value  = {}
        with open(dataset_file_path+".json") as json_file:
            d = json.load(json_file)
            for e in d.keys():
                for p in d[e]["prop"]:
                    if not dict_propTo_nameTo_value.get(p["name"]):
                        dict_propTo_nameTo_value[p["name"]] = {}
                    dict_propTo_nameTo_value[p["name"]][d[e]["name"]] = p["value"]
        return dict_propTo_nameTo_value

    #format {property : {entity : value}}
    d1_prop_entity_value = parse_json_dataset_to_dict(dataset1_file_path)
    d2_prop_entity_value = parse_json_dataset_to_dict(dataset2_file_path)

    dict_gold = {} #format {d1 : d2}
    with open(goldstandard_file_path, "r",encoding="UTF-8") as f:
        for line in f.read().split("\n"):
            if line:
                e1,e2 = line.split("\t")
                dict_gold[e1] = e2
    
    res = [] #format [[p1 (v1,v1'),(v2,v2')....],[p2 (v1,v1')...]
    for prop in d1_prop_entity_value:
        _res = []
        for e in d1_prop_entity_value[prop]:
            v1 = d1_prop_entity_value[prop][e]
            if (entity_d2 := dict_gold.get(e)):
                v2 = d2_prop_entity_value[prop][entity_d2]
                _res.append((v1[0],v2[0]))
        res.append(_res)
    return res

#keep only the mapped keys    
def mapped_key(keys_file_path_dataset1 : str, key_file_path_dataset2 : str) -> list[list[str]]:
    dict_keys = {}
    mapped_keys = []
    with open(keys_file_path_dataset1,"r",encoding="UTF-8") as f:
        for line in f.read().split("\n"):
            if line:
                prop_list = list(map(lambda x: x.strip(),line.split(",")))
                dict_keys[''.join(sorted(prop_list))] = 1

    with open(key_file_path_dataset2,"r",encoding="UTF-8") as f:
        for line in f.read().split("\n"):
            if line:
                prop_list = list(map(lambda x: x.strip(),line.split(",")))
                if dict_keys.get(''.join(sorted(prop_list))):
                    mapped_keys.append(prop_list)
    return mapped_keys

#check if key does not have empty values etc, filter out very bad keys
def not_empty_keys(key_list : list[list[str,str]], sparql_graph : str) -> list[list[str,str]]:
    """Test if a key is supported

    Args:
        key_list (list[list[str,str]]): _description_
        sparql_graph (str): _description_

    Returns:
        list[list[str,str]]: List of not empty keys
    """
    THRESHOLD_SUPPORT = 10
    not_empty_key_list = []
    for key in key_list:
        key_uri = [propName_to_propUri(p) for p in key]
        query = query_maker(key_uri, limit=100000, sparql_graph=sparql_graph,count=True)
        logging.info(f"Query for key :{key}, sparql : {query}")
        res = sparql_query(query)            
        if int(res["results"]["bindings"][0]["c"]['value']) > THRESHOLD_SUPPORT:  
            not_empty_key_list.append(key)
    return not_empty_key_list

#parse a vikey dataset and write a new file with uri for properties
def parse_vikey_dataset(dataset1_path : str, dataset2_path : str, dataset_class_name : str) -> None:
    def format_dataset(dataset_path : str, dataset_class_name :str, prefix_uri_dict : dict = "") -> None:
        new_prefix = dataset_class_name #add a prefix to every prop without prefix
        with open(dataset_path,"r",encoding="UTF-8") as f:
            with open(dataset_path+"_prefix",'w',encoding="UTF-8") as w:
                for line in f.read().split("\n"):
                    if line:
                        entity,prop,value = line.split("\t")
                        if ":" not in prop:
                            prop = new_prefix+":"+prop
                        w.write(entity+"\t"+prop+"\t"+value+"\n")                 

        #if we don't give a premade dict we assume this is the correct one to use
        if not prefix_uri_dict:
            #for every prefix we found we replace it with an uri
            prefix_uri = {
                dataset_class_name : "<http://simGen/"+dataset_class_name+"/",
                "skos"  : "<http://www.w3.org/2004/02/skos/core#"
            }

        with open(dataset_path+"_prefix","r",encoding="UTF-8") as f:
            with open(dataset_path+"_uri",'w',encoding="UTF-8") as w:
                for line in f.read().split("\n"):
                    if line:
                        entity,prop,value = line.split("\t")
                        prefix,name = prop.split(":")
                        prefix = prefix_uri[prefix]
                        w.write(entity+"\t"+prefix+name+">"+"\t"+value+"\n")

    format_dataset(dataset_path=dataset1_path, dataset_class_name=dataset_class_name)
    format_dataset(dataset_path=dataset2_path, dataset_class_name=dataset_class_name)

#find which prefix do we need to manually modif to a uri link
def find_unique_prefix_vikey(dataset_with_prefix : str) -> set[str]:
    unique_prefix = set()
    with open(dataset_with_prefix,"r",encoding="UTF-8") as f:
        for line in f.read().split("\n"):
            if line:
                entity,prop,value = line.split("\t")
                prefix,name = prop.split(":")
                unique_prefix.add(prefix)
    return unique_prefix

def is_file_empty(file_path : str) -> bool:

    path = pathlib.Path(file_path)
    f_size = path.stat().st_size
    return f_size < 500


#get all entities with all property they support
def get_entities_prop(sparql_graph : str, goldstandard_file_path : str) -> dict[str,list[str]]:
    query = """
    select distinct ?e ?p where {   
        graph <"""+sparql_graph+"""> {
            ?e ?p ?v.
            filter(isliteral(?v)) 
        }
    } 
    """
    goldstandard_entity_mapping = set()
    with open(goldstandard_file_path,"r", encoding="UTF-8") as f:
        for line in f.read().split("\n"):
            if line:
                e1,_ = line.split("\t")
                goldstandard_entity_mapping.add(e1)

    res = sparql_query(query)
    d = {}
    for line in res["results"]["bindings"]:
        entity_value = name_from_uri(line["e"]["value"])
        if entity_value in goldstandard_entity_mapping:
            prop_name = name_from_uri(line["p"]["value"])
            if not d.get(entity_value):
                d[entity_value] = set()
            d[entity_value].add(prop_name)
    for e in d:
        d[e] = list(d[e])
    return d


def get_keys_from_entity(entity_props :list[str], key_list : list[list[str]]) -> list[list[str]]:
    return np.array(key_list)[list(map(lambda x:not set(x).difference(set(entity_props)), key_list))].tolist()

#filter a dataset to get all entity that support the key
def filter_dataset(entity_props : dict[str,list[str]], keys : list[list[str]]):
    filtered_dataset = {}
    for k,v in entity_props.items():
        if all(list(map(lambda x : not set(x).difference(set(v)),keys))):
            filtered_dataset[k] = v
    return filtered_dataset

#from 2 two entities compute the similarity between them with a key and the associed treee
def compute_similarity(entity1 : dict[str,dict[str,str]], entity2 : dict[str,dict[str,str]], key : list[list[str]], tree_dict : dict[str,any], sparql_graph_names : tuple[str,str]) -> float:
    
    #compute value for every key, need to iter over keys
    def get_value(entity : str,prop: str, sparql_graph : str):
        prop_uri = propName_to_propUri(prop)
        #entity_uri = "/".join(prop_uri.split("/")[0:-1])+"/"+entity+">"
        #fix because sometimes with have a / at the begining
        query = """
        select ?v where {
            graph <"""+sparql_graph+"""> {
                bind(<"""+ entity +"""> as ?e).
                bind("""+ prop_uri+""" as ?p).
                ?e ?p ?v.
            }
        }
        """
        # logging.info(f"For prop : {prop}, sparql : {query}")
        res = sparql_query(query)
        value = ""
        for line in res["results"]["bindings"]:
            value+=line["v"]["value"]
        return value

    similarity = 0
    for prop in key:
        tree = tree_dict[prop]
        v1 = entity1[prop]
        v2 = entity2[prop]
        tree.set_leafs_value([v1,v2])
        similarity += tree.compute()
    similarity /= len(key)
    return similarity

def generate_dataset_entity(entity_uri :str, sparql_graph :str) -> dict[str,dict[str,str]]:
    """ Generate a sparql query to get all prop and values for an entity

    Args:
        entity_uri : uri of the entity
        sparql_graph (str): name of the sparql graph we want to query_

    Returns:
        str: return the sparql query in string format
    """
    query = """
    select * where {
        graph <"""+ sparql_graph +"""> {
            bind( <"""+ entity_uri +"""> as ?e).
            ?e ?p ?v
        }
    }
    """
    res = sparql_query(query)
    return res 

def full_dataset(sparql_graph :str) -> dict[str,dict[str,str]]:
    """ Generate a sparql query to get all prop and values for an entity

    Args:
        entity_uri : uri of the entity
        sparql_graph (str): name of the sparql graph we want to query_

    Returns:
        str: return the sparql query in string format
    """
    query = """
    select * where {
        graph <"""+ sparql_graph +"""> {
            ?e ?p ?v
        }
    }
    """
    entity_full_dataset = {}
    res = sparql_query(query)
    for line in res["results"]["bindings"]:
        entity_value = name_from_uri(line["e"]['value'])
        prop_name = name_from_uri(line["p"]["value"])
        # if "label" in prop_name:
        #     print(prop_name)
        prop_value = line["v"]["value"]
        if entity_value not in entity_full_dataset:
            entity_full_dataset[entity_value] = {}
        if not prop_name in entity_full_dataset[entity_value]:
            entity_full_dataset[entity_value][prop_name] = ""
        entity_full_dataset[entity_value][prop_name] += prop_value

    return entity_full_dataset


def test_stats(similarity_results_dict, goldstandard_file_path : str , threshold : float):
    mapped_entities = []    
    for e1 in similarity_results_dict:
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
    logging.info(f"For t = {threshold} \nf-measure : {f_measure} - precision : {precision} - recall : {recall} \n true positives : {tp} - false negatives : {fn} - true negatives : {tn} - false positives : {fp}")
    return precision,recall,f_measure




#####################################
########### Maybe useful ############
##################################### 

# supported_keys = lambda key_list, entity_props : np.array(key_list)[list(map(lambda x:not set(x).difference(set(entity_props)), key_list))].tolist()

# def _not_empty_keys(key_file_path : str, output_data_folder_path : str) -> list[list[str]]:
#     file_path_list = [] #save path to delete after testing
#     keys_list = []
#     with open(key_file_path, "r",encoding = "UTF-8") as f:
#         for line in f.read().split("\n"):
#             prop_list = list(map(lambda x: x.strip() if ":" not in x else x.strip().replace(":","_"),line.split(",")))        
#             prop_uri_list = list(map(lambda x : propName_to_propUri(x.strip()).strip(),prop_list))
#             file_path = output_data_folder_path+"_".join(prop_list)

#             keys_list.append(prop_list)
#             file_path_list.append(file_path)

#             get_dataset(prop_uri_list,file_path,limit=100)
#     correct_keys = []
#     for file_path,key in zip(file_path_list,keys_list):
#         if is_file_empty(file_path):
#             correct_keys.append(key)
#     return correct_keys

# #from a entity and a list of keys return all the keys supported
# def _get_keys_for_entity(entity : tuple[str,list[tuple[str,str]]], keys : list[list[str]]):
#     uri,prop_value = entity
    
#     #init key hash
#     dict_key = {}
#     #supported keys
#     support_keys = []

#     for k in keys:
#         dict_key[str(k)] = len(k)

#     for p,_ in prop_value:
#         for key in keys:
#             if p in key:
#                 dict_key[str(key)] -= 1
#                 if dict_key[str(key)] == 0:
#                     support_keys.append(key)
#     return support_keys

# def test_key(entity,key):
#     pass

# test_results = {}
# entities = get_entities()
# for entity in entities:
#     keys_supported = get_keys_for_entity(entity, key_list)
#     key_results = {}
#     for key in keys_supported:
#         results = test_key(entity,key)
#         key_results[str(key)] = results
#     test_results[str(key)] = deepcopy(key_results)

# #for all entity, test if couple have sim > threshold, if yes then true etc, compute recall, accuracy, f-measure for dataset
# def test_mapping_keys(entities : list[tuple[str,list[tuple[str,str]]]] ,keys):
#     pass
    
# def _run_simGen(key_file_path : str):
#     log.init()
#     if not os.path.isdir("img"):
#         subprocess.Popen(["mkdir","img"]).wait()


#  e2_full_dataset = {}
#     for e2 in E2:
#         res = tools.generate_dataset_entity(e2,DATASET2_graph)
#         e2_full_dataset[e2] = {}
#         for line in res["results"]["bindings"]:
#             prop_value = tools.name_from_uri(line["p"]["value"])
#             if not prop_value in e2_full_dataset[e2]:
#                 e2_full_dataset[e2][prop_value] = ""
#             e2_full_dataset[e2][prop_value] += line["v"]["value"]
