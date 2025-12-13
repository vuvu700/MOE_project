"""this implementation is based on the paper: 
Jacobs et al., Adaptive Mixtures of Local Experts, Neural Computation, 1991"""

import torch
import typing



class MOE_Model(torch.nn.Module):
    def __init__(
            self, experts:typing.Sequence[torch.nn.Module], gatingModel:torch.nn.Module,
            isClassif:bool, myLoss:bool, **kwargs) -> None:
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
        self.myLoss: bool = myLoss
        
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
        if self.myLoss is True:
            meanGate = gatingOutput.mean(dim=0)
            """shape: (nbExperts, ) mean gate activation of experts over the batch"""
            gateLoss = torch.pow(meanGate - float(1 / meanGate.shape[0]), 2.0).mean()
            loss += gateLoss
        return loss
    
    def applyLossCE(
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

