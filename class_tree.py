from typing import Callable
rootType = Callable[[str,str], float]
nodeType = Callable[[str], str]
leafType = Callable[[], str]

class SimTree():
    def __init__(self, value : leafType | nodeType | rootType, child : list):
        self.value = value
        self.child = child 

    def return_tree_asList(self):
        if len(self.child) == 2: #root 
            tree_list = [self.value,
                         self.child[0].return_tree_asList(),
                         self.child[1].return_tree_asList()]
        elif len(self.child) == 1: #nodes
            tree_list = [self.value,self.child[0].return_tree_asList()]
        else: #leaf
            tree_list = self.value 
        return tree_list
    
    def compute(self):
        if len(self.child) == 2: #root
            return self.value(self.child[0].compute(),
                              self.child[1].compute())
        elif len(self.child) == 1: #nodes
            return self.value(self.child[0].compute())
        else: #leaf
            return self.value()
        
    def __str__(self):
        if len(self.child) == 2: #root 
            tree_list = [self.value.__name__,
                         self.child[0].return_tree_asList,
                         self.child[1].return_tree_asList]
        elif len(self.child) == 1: #nodes
            tree_list = [self.value.__name__,self.child[0].return_tree_asList]
        else: #leaf
            tree_list = self.value 
        return str(tree_list)        
    
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






            
