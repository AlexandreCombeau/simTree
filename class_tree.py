from typing import Callable
import random

rootType = Callable[[str,str], float]
nodeType = Callable[[str], str]
leafType = str


class SimTree():

    def __init__(self, value : leafType | nodeType | rootType = None, child : list = None):
        self.value = value 
        self.child = child or []
        self.isRoot = lambda x: True if(len(x.child) == 2) else False
        self.isNode = lambda x: True if(len(x.child) == 1) else False
        self.isLeaf = lambda x: True if(len(x.child) == 0) else False
        
    def __str__(self):
        if self.isRoot(self): #root 
            tree_list = str([self.value.__name__,
                         self.child[0].__str__(),
                         self.child[1].__str__()])
        elif self.isNode(self): #nodes
            tree_list = [self.value.__name__,self.child[0].__str__()]
        else: #leaf
            tree_list = self.value
        return tree_list  
          
    def return_tree_asList(self):
        if self.isRoot(self): #root 
            tree_list = [self.value,
                         self.child[0].return_tree_asList(),
                         self.child[1].return_tree_asList()]
        elif self.isNode(self): #nodes
            tree_list = [self.value,self.child[0].return_tree_asList()]
        else: #leaf
            tree_list = self.value 
        return tree_list
      
    def compute(self):
        if self.isRoot(self): #root
            return self.value(self.child[0].compute(),
                              self.child[1].compute())
        elif self.isNode(self): #nodes
            return self.value(self.child[0].compute())
        else: #leaf
            return self.value
    
    def mutate(self, get_rd_function, probability : float):
        if not self.isLeaf(self):
            if random.random() < probability:
                self.value = get_rd_function(len(self.child),self.value)
            else:
                for c in self.child:
                    c.mutate(get_rd_function,probability)

    def set_leaf_value(self,value):
        if self.isNode(self):
            self.child[0].set_leaf_value(value)
        if self.isLeaf(self):
            self.value = value

    def set_leafs_value(self,values):
        x,y = values
        self.child[0].set_leaf_value(x)
        self.child[1].set_leaf_value(y)

    def get_Similarity_function(self):
        if self.isRoot(self): #root
            return self.value
        
    def get_transformations_functions(self):
        flatten = lambda l : [item for sublist in l for item in sublist]
        if self.isRoot(self) == 2: #root
            return flatten([self.child[0].get_transformations_functions,
                            self.child[1].get_transformations_functions])

        elif self.isNode(self) == 1: #nodes
            return self.value

    def find_depth(self,x):
        x+=1
        if self.isRoot(self):
            return max(self.child[0].find_depth(x), self.child[1].find_depth(x))
        if self.isNode(self):
            return self.child[0].find_depth(x)
        if self.isLeaf(self):
            return x    

    def get_depth(self):
        return self.find_depth(0)    


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