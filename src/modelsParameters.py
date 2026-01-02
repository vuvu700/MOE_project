
import torch
import torch.nn as nn
from typing import Tuple


def _count_parameters(module: nn.Module) -> Tuple[int, int]:
    """Count parameters directly owned by this module (not including children).
    Returns (total number of parameters, number of trainable parameters)"""
    total = 0
    trainable = 0
    for p in module.parameters(recurse=False):
        n = p.numel()
        total += n
        if p.requires_grad:
            trainable += n
    return total, trainable



class ModelTreeNode():
    def __init__(self, name:str, module:torch.nn.Module) -> None:
        self.name: str = name
        self.module: torch.nn.Module = module
        self.childrends: list[ModelTreeNode] = []
        self.__nbParamsCached: tuple[int, int]|None = None
        """(nbParams, nbTrainable) | None -> not yet computed"""
        
        allChildrens = list(module.named_children())
        for i, (child_name, child_module) in enumerate(allChildrens):
            if child_name.isdigit(): 
                child_name = f"{child_module._get_name()}[{child_name}]"
            self.childrends.append(ModelTreeNode(child_name, child_module))
        
    def nbParams(self, force:bool=False)->tuple[int, int]:
        if force or self.__nbParamsCached is None:
            newNbParams, newNbTrain = _count_parameters(self.module)
            for sub in self.childrends:
                delaParams, deltaTrain = sub.nbParams(force=force)
                newNbParams += delaParams
                newNbTrain += deltaTrain
            self.__nbParamsCached = (newNbParams, newNbTrain)
        return self.__nbParamsCached
                
    def print(
            self, prefix:str="", isLast:bool=True, 
            showTrainable:bool=True, end:tuple[str, ...]|None=None)->None:
        """prefix: Prefix used for tree formatting
        isLast: Whether this node is the last child
        showTrainable: Whether to display trainable parameters separately
        end: the start of the names to not sub develope more"""
        connector = "└── " if isLast else "├── "
        childPrefix = prefix + ("    " if isLast else "│   ")
        subtreeTotal, subtreeTrainable = self.nbParams()
        print(f"{prefix}{connector}{self.name}: {subtreeTotal:_} params" \
            + (f"({subtreeTrainable:_} trainable)" if showTrainable else ""))
        if (end is not None) and (self.name.startswith(end)):
            return

        iLast = (len(self.childrends) - 1)
        for i, sub in enumerate(self.childrends):
            sub.print(
                prefix=childPrefix, isLast=(i == iLast),
                showTrainable=showTrainable, end=end)
        
