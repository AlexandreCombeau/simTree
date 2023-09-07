import random
import numpy as np 
import tools
import logging
from copy import deepcopy
from simTree import SimTree,rootType,nodeType
from similarity import similarity_functions
from transformation import transformation_functions

           
flatten = lambda l : [item for sublist in l for item in sublist]
def get_rd_function(nb_child,value):
    if nb_child == 2:
        function_list = similarity_functions()
    else:
        function_list = transformation_functions()    
    flag = True
    while flag:
        new_value = random.choice(function_list)
        if new_value != value:
            flag = False
    return new_value

class SimGen():
    def __init__(self, param_algo : dict , param_data : dict):
        #main param
        self.population_size : int = param_algo["population_size"]
        self.nb_generation   : int = param_algo["nb_generation"]
        self.threshold       : float = param_algo["threshold"]
        #candidate param
        self.tree_max_depth  : int = param_algo["tree_max_depth"]
        #evolution param
        self.proba_selection : float = param_algo["proba_selection"]
        self.proba_mutation  : float = param_algo["proba_mutation"]
        self.proba_crossover : float = param_algo["proba_crossover"]
        self.selection_population_size   : int = round(self.proba_selection*self.population_size)
        self.mutation_population_size    : int = round(self.proba_mutation*self.population_size)
        self.crossover_population_size   : int = round(self.proba_crossover*self.population_size)
        self.random_tree_population_size : int = self.population_size - (
            self.mutation_population_size+
            self.crossover_population_size+
            self.selection_population_size
        ) #fill the population with random trees

        #init param
        self.population : list[SimTree] = [] #current population
        self.population_candidate : list[SimTree] = [] #candidate population from which we will create the next population
        self.population_scores : list[int] = [0]*self.population_size #every candidate has a score computed at the end of an generation
        self.population_similarity : list[int] = [0]*self.population_size
        self.similarity_functions : list[rootType] = param_data["similarity_functions"] 
        self.transformation_functions : list[nodeType] = param_data["transformation_functions"]

        #values param
        self.values : list[tuple[str,str]] = param_data["values"] #[(x1,y1),(x2,y2)...,(xn,yn)] such as (ei,p,xi) <=> (e'i,p',yi)
        self.negatives_samples : list[tuple[str,str]] = param_data['negatives_values']
        
        #meta param
        self.size_regularisation  = param_algo["size_regularisation"]
        self.negative_regularisation  = param_algo["negative_sampling_regularisation"]
        self.gen_scores = []
        self.gen_sim = []
        self.gen_top_k_sim = []
        self.gen_top_k_score = []
        self.top_k = param_algo["top_k"]
        self.freq_tf = []
        self.freq_sf = []
        self.size_tracker = []

        #logging param
        self.param_algo = param_algo
        self.param_data = param_data
        
    def generate_random_tree(self):
        nb_transformation = random.randint(1,self.tree_max_depth - 2)
        tf_left = np.random.choice(self.transformation_functions,nb_transformation,replace=False)
        tf_left = np.append(tf_left,["a"]) #TODO check comment mieux gerer le lift
        tf_right = np.random.choice(self.transformation_functions,nb_transformation,replace=False)
        tf_right = np.append(tf_right,["a"])
        sf = np.random.choice(self.similarity_functions,1)[0]
        def nest_list(lst):
            #format our list of function : [f,g,h,...]
            #into : [f,[g,[h,[...]]]]
            if len(lst) == 1:
                return [lst[0]]
            else:
                return [lst[0], nest_list(lst[1:])]

        tree_list = [sf,nest_list(tf_left),nest_list(tf_right)]
        tree = tools.tree_from_list(tree_list)
        return tree

    def init_population(self):
        """
            Generate a population of size n with random tree
        
        """
        #check variance and stuff meta data TODO
        for _ in range(self.population_size):
            self.population.append(self.generate_random_tree())
    
   
    def fitness_candidate(self,candidate : SimTree) -> tuple[float,float]:
        sim_score = 0
        for values in self.values:
            #compute score for each value pair for one tree
            candidate.set_leafs_value(values)
            sim_score += candidate.compute()
        sim_score /= len(self.values)


        negative_sim_score = 0
        if self.negatives_samples:
            for negatives_values in self.negatives_samples:
                candidate.set_leafs_value(negatives_values)
                negative_sim_score += candidate.compute()
            negative_sim_score /= len(self.negatives_samples)

        #TODO maybe use other metrics
        #depth scoring
        min_depth = 3
        #depth_score = (self.tree_max_depth - min_depth)/candidate.get_depth()
        depth_score = (candidate.get_depth() - min_depth)/(self.tree_max_depth - min_depth)
        fitness_score = sim_score - ((self.size_regularisation*depth_score) + (self.negative_regularisation*negative_sim_score))

        return fitness_score, sim_score

    
    def fitness_over_population(self) -> None:
        '''
            Compute the score of a tree on every value pair
            Keep the mean as a score
        '''    
        for index,candidate in enumerate(self.population):
            fitness,similarity = self.fitness_candidate(candidate)
            self.population_scores[index] = fitness
            self.population_similarity[index] = similarity
                                    

    def tournament_selection(self):
        while len(self.population_candidate) < self.selection_population_size:
            index_candidate1, index_candidate2 = np.random.choice(np.arange(0,len(self.population)),2,replace=False)
            if self.population_scores[index_candidate1] > self.population_scores[index_candidate2]:
                self.population_candidate.append(deepcopy(self.population[index_candidate1]))
            else:
                self.population_candidate.append(deepcopy(self.population[index_candidate2]))

    def mutation(self):
        mutation_candidate = []
        while len(mutation_candidate) < self.mutation_population_size:
            mutation_candidate.append(np.random.choice(self.population_candidate,1)[0])
        
        for candidate in mutation_candidate:
            root_pointer = deepcopy(candidate)
            tree = root_pointer
            invalid_mutation = True
            
            while invalid_mutation:
                mutation_type = np.random.choice(["add","modify","remove"])
                if mutation_type =="remove":
                    #if condition not correct we loop because we cannot remove every transformation function
                    invalid_mutation = tree.child[0].get_depth() <= 2 and tree.child[1].get_depth() <= 2
                elif mutation_type == "add":
                    invalid_mutation = tree.child[0].get_depth() >= self.tree_max_depth - 2 and tree.child[1].get_depth() >= self.tree_max_depth - 2
                else:  #valid mutation
                    invalid_mutation = False
            match mutation_type:
                case "modify":
                    if np.random.choice(['root','nodes']) == 'nodes':
                        #print("change node")
                        tree = np.random.choice(tree.child) #choose left or right branche
                        max_depth = tree.get_depth() - 1 #minus leaf, 0 is current node, 1 is next node ... etc
                        depth_choice = np.random.choice(max_depth,1)[0]
                        while depth_choice > 0:
                            tree = tree.child[0]
                            depth_choice -=1
                        #depth_choice should be 0
                    #else:
                        #print("change root")
                    tree.value = get_rd_function(len(tree.child),tree.value)

                case "add":
                    child_choice = np.random.choice([0,1]) #choose left or right branche
                    if tree.child[0].get_depth() >= self.tree_max_depth - 1:
                        tree = tree.child[1]
                        child_choice = 1
                    elif tree.child[1].get_depth() >= self.tree_max_depth - 1:
                        tree = tree.child[0]
                        child_choice = 0
                    else:                
                        tree = tree.child[child_choice]
                    max_depth = tree.get_depth()
                    target_depth = np.random.choice(max_depth,1)[0]
                    #print("target_depth",target_depth)
                    #print("max depth", tree.get_depth())
                    before = 0 #we need to insert a node before the current one, i.e : a new node conected to the root
                    after  = 1 # we keep track of depths, we stop if we find our target depth
                    while before != target_depth and after != target_depth: #if after find target or before already is target
                        tree = tree.child[0]
                        after  += 1
                    #we found our target depth
                    #print(f"after : {after}, before {before}")
                    if before == target_depth: #we need to add a node connected to the root
                        _tmp = root_pointer.child[child_choice]
                        root_pointer.child[child_choice] = SimTree(get_rd_function(len(_tmp.child),None), [_tmp])
                    if after == target_depth: #we add a new node after the current
                        _tmp = tree.child[0]
                        tree.child[0] = SimTree(get_rd_function(len(_tmp.child),None),[_tmp])

                case "remove":
                    child_choice = np.random.choice([0,1]) #choose left or right branche
                    if tree.child[0].get_depth() <= 2:
                        tree = tree.child[1]
                        child_choice = 1
                    elif tree.child[1].get_depth() <=2:
                        tree = tree.child[0]
                        child_choice = 0
                    else:          
                        tree = deepcopy(tree.child[child_choice])

                    max_depth = tree.get_depth() - 1
                    target_depth = np.random.choice(max_depth,1)[0]
                    current_depth = target_depth
                    while current_depth != 0:
                        before_pointer = tree
                        tree = tree.child[0]
                        current_depth -=1
                        #we found our target depth
                        #print(f"target_depth : {target_depth}")
                        #link previous child we after child
                    if target_depth == 0:
                        root_pointer.child[child_choice] = deepcopy(tree.child[0])
                    else:
                        before_pointer.child[0] = deepcopy(tree.child[0])
            
            self.population_candidate.append(root_pointer) #append new created element

    def crossover(self):
        crossover_candidates = []
        while len(crossover_candidates) < self.crossover_population_size:
            crossover_candidates.append(np.random.choice(self.population_candidate,2))

        for t1,t2 in crossover_candidates:
            node_choice = [
                [deepcopy(t1.child[0]), deepcopy(t2.child[1])],
                [deepcopy(t2.child[0]), deepcopy(t1.child[1])],
            ]
            new_root_value = np.random.choice([t1.value,t2.value])
            new_tree = SimTree(new_root_value,node_choice[np.random.choice(2)])
            self.population_candidate.append(new_tree)

    def random_selection(self):
        random_trees = []
        while len(random_trees) < self.random_tree_population_size:
            random_trees.append(self.generate_random_tree())      
        self.population_candidate = np.append(self.population_candidate,random_trees)
        
    def stats(self):
        self.gen_sim.append(deepcopy(self.population_similarity))
        self.gen_scores.append(deepcopy(self.population_scores))
        self.gen_top_k_sim.append(deepcopy(sorted(self.population_similarity,reverse=True)[:self.top_k]))
        self.gen_top_k_score.append(deepcopy(sorted(self.population_scores,reverse=True)[:self.top_k]))
        self.size_tracker.append([t.get_depth() for t in self.population])

        #sim and trans functions frequency and evolution tracker, we want to analyse the distribution of functions choice over generations
        freq_trans = {}
        freq_sim   = {}
        for f in transformation_functions():
            freq_trans[f.__name__] = 0 

        for s in similarity_functions():
            freq_sim[s.__name__] = 0

        for tree in self.population:
            left, right = tree.get_transformations_functions()
            for f in left+right:
                freq_trans[f.__name__] +=1
            freq_sim[tree.get_similarity_function().__name__] +=1
            
        self.freq_sf.append(freq_sim)
        self.freq_tf.append(freq_trans)

    
    def log(self):
        logging.info(f"Parameters used : \n{self.param_algo} \n\t")
        best_score_tree = max(self.population_scores)
        best_sim_tree = max(self.population_similarity)
        logging.info(f"\n Best similarity tree :  {best_sim_tree} \t Best scoring tree : {best_score_tree}")

    def evolve_population(self):
        #compute score for each tree
        self.init_population()
        for _ in range(self.nb_generation):

            self.fitness_over_population()
            self.tournament_selection()
            self.crossover()
            self.mutation()
            self.random_selection()

            self.population = self.population_candidate
            self.population_candidate = []
            self.stats()
 

        self.fitness_over_population()
        self.stats()   
        self.log()  

