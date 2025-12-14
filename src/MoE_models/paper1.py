"""this implementation is based on the paper: 
Jacobs et al., Adaptive Mixtures of Local Experts, Neural Computation, 1991"""

import torch
import typing

_version = typing.Literal["original", "originalWithCE", "logLikelihood", ]

class MOE_Model(torch.nn.Module):
    def __init__(
            self, experts:typing.Sequence[torch.nn.Module], gatingModel:torch.nn.Module,
            isClassif:bool, loadBalance:bool, useVersion:_version, **kwargs) -> None:
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
        self.lossVersion: _version = useVersion
        
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
    
    def applyLossAll(
            self, mergedExpertsOutputs:torch.Tensor, 
            gatingOutput:torch.Tensor, truth:torch.Tensor)->torch.Tensor:
        kwargs = {"mergedExpertsOutputs": mergedExpertsOutputs, "gatingOutput": gatingOutput}
        if self.isClassif is True:
            if self.lossVersion == "original":
                loss = self._applyOriginalLoss_classif(**kwargs, truthLabels=truth)
            elif self.lossVersion == "originalWithCE":
                loss = self._applyLossCE(**kwargs, truthLabels=truth)
            elif self.lossVersion == "logLikelihood":
                loss = self._applyLossLogLikelihood(**kwargs, truthLabels=truth)
            else: raise ValueError(f"invalide loss version with classification: {self.lossVersion}")
        else: loss = self._applyOriginalLoss(**kwargs, truth=truth)
        return self._addExpertsBalancingLoss(loss=loss, gatingOutput=gatingOutput)
        
    def _addExpertsBalancingLoss(
            self, loss:torch.Tensor, gatingOutput:torch.Tensor)->torch.Tensor:
        """conditionaly add the experts load balancing loss if needed\n
        loss: (1, )\n
        gatingOutput: (batchSize, nbExperts)"""
        if self.loadBalance is True: # 
            meanGate = gatingOutput.mean(dim=0)
            """shape: (nbExperts, ) mean gate activation of experts over the batch"""
            balancingLoss = torch.pow(meanGate - float(1 / meanGate.shape[0]), 2.0).mean()
            loss = loss + balancingLoss
        return loss
    
    def _applyOriginalLoss(
            self, mergedExpertsOutputs:torch.Tensor, 
            gatingOutput:torch.Tensor, truth:torch.Tensor)->torch.Tensor:
        """compute the original loss from the paper, that is meant for regression\n
        `mergedExpertsOutputs`: (batchSize, nbOuts, nbExperts)\n
        `gatingOutput`: (batchSize, nbExperts)\n
        `truthLabels`: (batchSize, nbOuts)\n
        return the loss of the batch"""
        expertsLoss = torch.norm((truth - mergedExpertsOutputs), p=2, dim=1) ** 2
        """shape: (batchSize, nbExperts)"""
        return self.__originalGenericApplyLoss(expertsLoss=expertsLoss, gatingOutput=gatingOutput)
    
    def __originalGenericApplyLoss(
            self, expertsLoss:torch.Tensor, gatingOutput:torch.Tensor)->torch.Tensor:
        """internal methode to compute the loss on each experts loss, 
            based on the equation 1.3 from the paper. (apply mean reduction)\n
        `expertsLoss`: (batchSize, nbExperts)\n
        `gatingOutput`: (batchSize, nbExperts)\n"""
        #loss = ((gatingOutput * expertsLoss).sum(dim=1)).mean() # first proposition from the paper
        loss = - torch.log((gatingOutput * torch.exp(-0.5 * expertsLoss)).sum(dim=1)).mean()
        return loss
    
    def _applyLossCE(
            self, mergedExpertsOutputs:torch.Tensor, 
            gatingOutput:torch.Tensor, truthLabels:torch.Tensor)->torch.Tensor:
        """the adapted loss from the paper to use CE insted of the MSE.
        `mergedExpertsOutputs`: (batchSize, nbClasses, nbExperts)\n
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
        return self.__originalGenericApplyLoss(
            expertsLoss=expertsCE, gatingOutput=gatingOutput)

    def _applyOriginalLoss_classif(
            self, mergedExpertsOutputs:torch.Tensor, 
            gatingOutput:torch.Tensor, truthLabels:torch.Tensor)->torch.Tensor:
        """compute the original loss from the paper and \
            just convert the truthLabels to the expected probability distribution\n
        `mergedExpertsOutputs`: (batchSize, nbClasses, nbExperts)\n
        `gatingOutput`: (batchSize, nbExperts)\n
        `truthLabels`: (batchSize, ) with the index of the label to predict"""
        (batchSize, nbClasses, nbExperts) = mergedExpertsOutputs.shape
        flatTruthClasses = torch.zeros((batchSize, nbClasses, nbExperts), device=truthLabels.device)
        flatTruthClasses[torch.arange(batchSize, device=truthLabels.device), truthLabels] = 1.0
        """shape: (batchSize, nbClasses, nbExperts)"""
        mergedExpertsOutputs = torch.softmax(mergedExpertsOutputs, dim=1)
        return self._applyOriginalLoss(
            mergedExpertsOutputs=mergedExpertsOutputs, 
            gatingOutput=gatingOutput, truth=flatTruthClasses)
    
    def _applyLossLogLikelihood(
            self, mergedExpertsOutputs:torch.Tensor, 
            gatingOutput:torch.Tensor, truthLabels:torch.Tensor)->torch.Tensor:
        """a better adaptation of the loss from the paper to classification\n
        `mergedExpertsOutputs`: (batchSize, nbClasses, nbExperts)\n
        `gatingOutput`: (batchSize, nbExperts)\n
        `truthLabels`: (batchSize, ) with the index of the label to predict"""
        (batchSize, nbClasses, nbExperts) = mergedExpertsOutputs.shape
        batchIter = torch.arange(batchSize, device=mergedExpertsOutputs.device)
        expertsOutsProba = torch.softmax(mergedExpertsOutputs, dim=1)
        """shape: (batchSize, nbClasses, nbExperts) == p_i"""
        targetedProbas = expertsOutsProba[batchIter, truthLabels, :]
        """shape: (batchSize, nbExperts) == p_{i,y}"""
        targetedProbas *= gatingOutput
        """== g_i * p_{i,y}"""
        return - torch.log(targetedProbas.sum(dim=1)).mean()