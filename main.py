import numpy as np 
from functools import reduce 
from similarity import *
from transformation import *
from copy import deepcopy
import tools
import simGen
#TODO extract param to name img
def genetic_algo(dataset : list[tuple[str,str]], nb_positives : int, negative_sampling : bool = False, param_algo : dict = None, plot_img_path : str = ""):
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
        negative_param_algo = deepcopy(param_algo)
        negative_param_algo["nb_generation"] = 5
        neg_sim = simGen.SimGen(negative_param_algo,param_data)
        neg_sim.evolve_population()
        supposed_best_sim_function = get_tf_function_from_name(max(neg_sim.freq_tf[-1])) #get most frequent sim function for the last generation
        negatives = tools.generate_NegativeSample(list(map(lambda x: [x] ,dataset)),supposed_best_sim_function)
        negative_dataset = np.array(negatives)[np.random.choice(np.arange(0,len(negatives)),nb_negative_example)].tolist()
        param_data["negatives_values"] = negative_dataset        

    sim = simGen.SimGen(param_algo,param_data)
    sim.evolve_population()
    tools.plot_stats_sim(sim, plot_img_path)
    best_trees = tools.get_best_tree(sim)

    return sim, best_trees, index_positives






def main():
    N = 1000
    isbn_dataset = tools.generate_dataset(db_prop="isbn",wk_prop="P957",size=N)
    sim_isbn, best_trees_isbn, index_positives_isbn = genetic_algo(isbn_dataset,100, True, "img/isbn/100pop")

    release_dataset = tools.generate_dataset(db_prop="releaseDate",wk_prop="P577",size=N)
    sim_releaseDate, best_trees_releaseDate, index_positives_releaseDate = genetic_algo(release_dataset,100,50,"img/release_date/100pop")
    #remove index_positives element from the dataset
    #need to be the same (maybe take the same seed)
    acc,recall = tools.test_solution(sim_trees=[best_trees_isbn[0],best_trees_releaseDate[0]],dataset=[isbn_dataset,release_dataset],threshold=0.7)
    print(f"accuracy = {acc}, recall = {recall}")


main()