"""this implementation is based on the paper: 
Jacobs et al., Adaptive Mixtures of Local Experts, Neural Computation, 1991"""

import torch
import typing

from torch.nn.modules.module import Module



class MOE_Model(torch.nn.Module):
    def __init__(
            self, experts:typing.Sequence[torch.nn.Module], gatingModel:torch.nn.Module,
            isClassif:bool, loadBalance:bool, useOriginal:bool, **kwargs) -> None:
        """create the MoE model with the given experts and gating\n
        `experts` and `gatingModel` must take the same input, 
        `experts` must all give the same output: (batchSize, nbOuts),
        the `gatingModel` must give: (batchSize, nbExperts)"""
        super().__init__(**kwargs)
        self.isClassif: bool = isClassif
        self.experts:list[torch.nn.Module] = []
        self.gatingModel: torch.nn.Module = gatingModel
        for i, expert in enumerate(experts):
            self.add_module(f"expert[{i}]", expert)
            self.experts.append(expert)
        self.loadBalance: bool = loadBalance
        self.useOriginal: bool = useOriginal
        
    @property
    def nbExperts(self)->int:
        return len(self.experts)
    
    def __call__(self, x:torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return super().__call__(x)
    
    def forward(self, x:torch.Tensor)->tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """return (MoE_mergedOutput, outputsOfEachExperts, gatingOfExperts[softmaxed])
         - x must be of shape (batchSize, ...)
         - MoE_mergedOutput: (batchSize, nbOutputs)
         - outputsOfEachExperts: (batchSize, nbExperts, nbOutputs)
         - gatingOfExperts: (batchSize, nbExperts)"""
        gatingOutput: torch.Tensor = \
            torch.softmax(self.gatingModel(x), dim=1)
        """shape: (batchSize, nbExperts)"""
        assert gatingOutput.shape[1] == self.nbExperts, \
            f"gating network outputed the wrong nb of weights, " \
            f"got {gatingOutput.shape[1]}, expected {self.nbExperts}"
        expertsOutputs: list[torch.Tensor] = [
            expert(x) for expert in self.experts]
        """list of tensors of shape: (batchSize, nbOutputs)"""
        mergedExpertsOutputs = torch.stack(expertsOutputs, dim=2)
        """shape: (batchSize, nbOutputs, nbExperts)"""
        mergedOutput = self.merge(mergedExpertsOutputs, gatingOutput)
        """shape: (batchSize, nbOutputs)"""
        return (mergedOutput, mergedExpertsOutputs, gatingOutput)
    
    def merge(self, expertsLogits:torch.Tensor, gatingOutput:torch.Tensor)->torch.Tensor:
        """return the merged output of the experts (batchSize, nbOutputs)\n
        expertsLogits: (batchSize, nbOutputs, nbExperts)\n
        gatingOutput: (batchSize, nbExperts)"""
        if self.isClassif is True:
            expertsLogits = torch.softmax(expertsLogits, dim=1)
        return (gatingOutput[:, None, :] * expertsLogits).sum(dim=2)
    
    def _genericApplyLoss(self, expertsLoss:torch.Tensor, gatingOutput:torch.Tensor)->torch.Tensor:
        """return the loss with the generic formula computed\
         for each element of the batch (don't reduce)\n
        `expertsLoss`: (batchSize, nbExperts)\n
        `gatingOutput`: (batchSize, nbExperts)\n"""
        loss = - torch.log((gatingOutput * torch.exp(-0.5 * expertsLoss)).sum(dim=1)).mean()
        if self.loadBalance is True:
            meanGate = gatingOutput.mean(dim=0)
            """shape: (nbExperts, ) mean gate activation of experts over the batch"""
            gateLoss = torch.pow(meanGate - float(1 / meanGate.shape[0]), 2.0).mean()
            loss += gateLoss
        return loss
    
    def applyLossAll(
            self, mergedExpertsOutputs:torch.Tensor, 
            gatingOutput:torch.Tensor, truth:torch.Tensor)->torch.Tensor:
        kwargs = {"mergedExpertsOutputs": mergedExpertsOutputs, "gatingOutput": gatingOutput}
        if self.isClassif is True:
            if self.useOriginal is True:
                return self._applyOriginalLoss_classif(**kwargs, truthLabels=truth)
            else: return self._applyLossCE(**kwargs, truthLabels=truth)
        else: return self._applyOriginalLoss(**kwargs, truth=truth)
    
    def _applyLossCE(
            self, mergedExpertsOutputs:torch.Tensor, 
            gatingOutput:torch.Tensor, truthLabels:torch.Tensor)->torch.Tensor:
        """`mergedExpertsOutputs`: (batchSize, nbClasses, nbExperts)\n
        `gatingOutput`: (batchSize, nbExperts)\n
        `truthLabels`: (batchSize, ) with the index of the label to predict\n
        return the loss of the batch"""
        (batchSize, nbClasses, nbExperts) = mergedExpertsOutputs.shape
        flatExpertsOutputs = torch.transpose(
            mergedExpertsOutputs, 1, 2).reshape(batchSize * nbExperts, nbClasses)
        """shape: (batchSize * nbExperts, nbClasses)"""
        flatTruthLabels = truthLabels.view((-1, 1)).expand((-1, nbExperts)).flatten()
        """shape: (batchSize * nbExperts, )"""
        flatExpertsCE = torch.nn.functional.cross_entropy(
            flatExpertsOutputs, flatTruthLabels, reduction="none")
        """shape: (batchSize * nbExperts)"""
        expertsCE = flatExpertsCE.reshape(batchSize, nbExperts)
        """shape: (batchSize, nbExperts)"""
        return self._genericApplyLoss(
            expertsLoss=expertsCE, gatingOutput=gatingOutput)

    def _applyOriginalLoss(
            self, mergedExpertsOutputs:torch.Tensor, 
            gatingOutput:torch.Tensor, truth:torch.Tensor)->torch.Tensor:
        """`mergedExpertsOutputs`: (batchSize, nbOuts, nbExperts)\n
        `gatingOutput`: (batchSize, nbExperts)\n
        `truthLabels`: (batchSize, nbOuts)\n
        return the loss of the batch"""
        expertsLoss = torch.norm((truth - mergedExpertsOutputs), p=2, dim=1) ** 2
        """shape: (batchSize, nbExperts)"""
        return self._genericApplyLoss(expertsLoss=expertsLoss, gatingOutput=gatingOutput)

    def _applyOriginalLoss_classif(
            self, mergedExpertsOutputs:torch.Tensor, 
            gatingOutput:torch.Tensor, truthLabels:torch.Tensor)->torch.Tensor:
        """`mergedExpertsOutputs`: (batchSize, nbClasses, nbExperts)\n
        `gatingOutput`: (batchSize, nbExperts)\n
        `truthLabels`: (batchSize, ) with the index of the label to predict\n
        return the loss of the batch"""
        (batchSize, nbClasses, nbExperts) = mergedExpertsOutputs.shape
        flatTruthClasses = torch.zeros((batchSize, nbClasses, nbExperts), device=truthLabels.device)
        flatTruthClasses[torch.arange(batchSize, device=truthLabels.device), truthLabels] = 1.0
        """shape: (batchSize, nbClasses, nbExperts)"""
        mergedExpertsOutputs = torch.softmax(mergedExpertsOutputs, dim=1)
        return self._applyOriginalLoss(
            mergedExpertsOutputs=mergedExpertsOutputs, 
            gatingOutput=gatingOutput, truth=flatTruthClasses)
    

"""
tensor([7, 4, 9, 8, 8, 3, 2, 0, 1, 2, 6, 9, 0, 2, 2, 6, 5, 7, 9, 3, 5, 9, 0, 3,
        4, 9, 3, 4, 5, 5, 1, 8, 1, 9, 9, 1, 8, 0, 1, 4, 3, 6, 1, 7, 7, 3, 9, 9,
        5, 2, 9, 5, 5, 9, 9, 0, 5, 7, 5, 0, 2, 5, 1, 2, 9, 6, 9, 5, 6, 4, 2, 0,
        6, 8, 3, 7, 9, 7, 2, 1, 9, 9, 7, 5, 3, 0, 3, 1, 9, 9, 8, 4, 0, 6, 9, 9,
        8, 5, 1, 0, 5, 1, 8, 4, 1, 7, 1, 8, 7, 1, 6, 2, 4, 3, 8, 9, 3, 9, 7, 7,
        9, 0, 0, 7, 5, 1, 1, 3, 6, 8, 3, 3, 0, 7, 2, 3, 4, 1, 5, 6, 9, 6, 2, 8,
        7, 8, 3, 6, 1, 1, 8, 6, 1, 3, 9, 2, 8, 7, 7, 9, 8, 2, 2, 1, 2, 9, 7, 0,
        2, 4, 8, 8, 2, 3, 1, 5, 9, 3, 5, 0, 6, 0, 1, 5, 7, 6, 8, 3, 5, 0, 3, 2,
        4, 2, 6, 4, 1, 1, 5, 7, 7, 1, 8, 4, 4, 6, 3, 2, 8, 0, 2, 2, 0, 4, 1, 8,
        0, 9, 1, 5, 0, 7, 8, 9, 0, 4, 6, 2, 0, 7, 6, 7, 2, 5, 1, 2, 8, 0, 2, 2,
        2, 0, 0, 4, 5, 3, 8, 9, 7, 0, 3, 0, 0, 9, 5, 4], device='cuda:0')
"""