"""this implementation is based on the paper:
Jacobs et al., Adaptive Mixtures of Local Experts, Neural Computation, 1991"""

import torch
import typing
import attrs



class Block_CNN(torch.nn.Module):
    def __init__(
            self, nbIn:int, nbOut:int,
            usePadding:bool, useBatchNorm:bool, usePool:bool) -> None:
        super().__init__()
        self.nbIn: int = nbIn
        self.nbOut: int = nbOut
        self.cnn = torch.nn.Conv2d(
            nbIn, nbOut, kernel_size=3, bias=False,
            padding=("same" if usePadding else 0))
        self.norm = (torch.nn.BatchNorm2d(nbOut) if useBatchNorm else None)
        self.relu = torch.nn.ReLU()
        self.pool = (torch.nn.MaxPool2d(2) if usePool else None)

    def __call__(self, x:torch.Tensor) -> torch.Tensor:
        return super().__call__(x)

    def forward(self, x:torch.Tensor)->torch.Tensor:
        """x: (batchSize, nbChannels, H, W)"""
        x = self.cnn(x)
        if self.norm is not None:
            x = self.norm(x)
        x = self.relu(x)
        if self.pool is not None:
            x = self.pool(x)
        return x


class GatingWithNoise(torch.nn.Module):
    def __init__(self, nbIn:int, nbExperts:int) -> None:
        super().__init__()
        self.gate = torch.nn.Linear(nbIn, nbExperts)
        self.noise = torch.nn.Linear(nbIn, nbExperts)
        
    def __call__(self, x:torch.Tensor)->torch.Tensor:
        return super().__call__(x)
    
    def forward(self, x:torch.Tensor)->torch.Tensor:
        """x: (batch, features)
        return (batch, experts)"""
        gateOut = self.gate(x)
        noiseOut = torch.nn.functional.softplus(self.noise(x))
        trueNoise = torch.randn_like(noiseOut)
        return gateOut + (trueNoise * noiseOut)
        
class Block_FFD_Moe(torch.nn.Module):
    def __init__(
            self, nbIn:int, nbOut:int, nbExperts:int,
            topK:int, fromCNN:bool, memMode:bool=False) -> None:
        super().__init__()
        self.topK: int = topK
        self.fromCNN: bool = fromCNN
        self.memEfficientMode: bool = memMode
        self.nbOut: int = nbOut
        self.gating = GatingWithNoise(nbIn, nbExperts)
        nbHidden = max(nbIn, nbOut) * 4
        self.experts = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Linear(nbIn, nbHidden), torch.nn.ReLU(),
                torch.nn.Linear(nbHidden, nbOut))
            for _ in range(nbExperts)])

    @property
    def nbExperts(self)->int:
        return len(self.experts)

    def __call__(self, x:torch.Tensor)->tuple[torch.Tensor, torch.Tensor]:
        return super().__call__(x)
    
    def forward(self, x:torch.Tensor)->tuple[torch.Tensor, torch.Tensor]:
        """x: [normal](batch, features) | [CNN](batch, channel, H, W)"""
        if self.fromCNN:
            batchSize, nbFeatures, H, W = x.shape
            x = x.permute((0, 2, 3, 1)) # (B, H, W, C)
            x = x.reshape((batchSize*H*W), nbFeatures) # (B * H * W, C)
        else:
            H, W = (0, 0)
            batchSize, nbFeatures = x.shape
        # => x: (batch, features)
        batchSize2, nbFeatures = x.shape
        
        gatingLogits = self.gating(x) # (batch, experts)
        gatingLogits_k, topK_indices = torch.topk(gatingLogits, k=self.topK, dim=1)
        # (batch, k), (batch, k)
        gatingProb = torch.softmax(gatingLogits_k, dim=1) # (batch, k)
        allExperts_weigths = torch.zeros((batchSize2, self.nbExperts), device=x.device) # (batch, experts)
        allExperts_weigths.scatter_(dim=1, index=topK_indices, src=gatingProb)

        if self.memEfficientMode is True:
            xOut = torch.zeros((batchSize2, self.nbOut), device=x.device)
            for iExpert in range(self.nbExperts):
                expertUse, *_ = torch.where((topK_indices == iExpert).any(dim=1))
                if expertUse.shape[0] == 0:
                    continue # expert isn't used
                expertOuts = self.experts[iExpert](x[expertUse])
                expertGates = allExperts_weigths[expertUse, iExpert].unsqueeze(dim=1)
                res = (expertGates * expertOuts)
                xOut.index_add_(dim=0, index=expertUse, source=res)
        else: 
            expertsOut = torch.stack([expert(x) for expert in self.experts], dim=2)
            xOut = torch.sum(allExperts_weigths[:, None, :] * expertsOut, dim=2)
        
        if self.fromCNN:
            xOut = xOut.reshape(batchSize, H, W, self.nbOut) # (B, H, W, C)
            xOut = xOut.permute(0, 3, 1, 2) # (B, C, H, W)
            allExperts_weigths = allExperts_weigths.reshape(batchSize, H, W, self.nbExperts) # (B, H, W, experts)
        return (xOut, allExperts_weigths)





class VisionModelMoe(torch.nn.Module):
    def __init__(self, blocks:list[torch.nn.Module], nbClasses:int, 
                 wImp:float, wLoad:float, **kwargs) -> None:
        """..."""
        super().__init__(**kwargs)
        self.blocks = torch.nn.ModuleList(blocks)
        self.nbClasses: int = nbClasses
        self.wImportance: float = wImp
        self.wLoad: float = wLoad


    def __call__(self, x:torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        return super().__call__(x)

    def forward(self, x:torch.Tensor)->tuple[torch.Tensor, list[torch.Tensor]]:
        """return (MoE_mergedOutput, outputsOfEachExperts, gatingOfExperts[softmaxed])
         - x must be of shape (batchSize, ...)
         - MoE_mergedOutput: (batchSize, nbOutputs)
         - outputsOfEachExperts: (batchSize, nbExperts, nbOutputs)
         - gatingOfExperts: (batchSize, nbExperts)"""
        allGates = []
        for block in self.blocks:
            if isinstance(block, Block_FFD_Moe):
                x, gates = block(x)
                allGates.append(gates)
            else: x = block(x)
        return x, allGates

    @staticmethod
    def cv_squared(x:torch.Tensor, eps:float=1e-10) ->torch.Tensor:
        """squared coeff of variation: var(x) / mean(x)^2"""
        mean = x.mean()
        var = x.var(unbiased=False)
        return var / (mean ** 2 + eps)

    def balanceLoss(self, gatings:list[torch.Tensor]):
        """`gatings`: list[(batch, experts)]"""
        totalLoss = torch.tensor(0) # (1, )
        for gate in gatings:
            importance = gate.sum(dim=0) # (experts, )
            lossImportance = self.wImportance * self.cv_squared(importance) # (1, )
            load = load = (gate > 0).float().sum(dim=0) # (experts, )
            lossLoad = self.wLoad * self.cv_squared(load) # (1, )
            totalLoss = totalLoss + (lossImportance + lossLoad) # (1, )
        return totalLoss # (1, )

    @staticmethod
    def get_cifar_v1(
            nbClasses:int, nbExperts:int, topK:int,
            modelConfig:typing.Literal["small", "medium", "large"],
            wImp:float, wLoad:float, memoryMode:bool, dropout=0.25)->"VisionModelMoe":
        """the V1 will use MOE at the end of the network to output the logits"""
        if modelConfig == "small":
            return VisionModelMoe(nbClasses=nbClasses, wImp=wImp, wLoad=wLoad, blocks=[
                Block_CNN(nbIn=3, nbOut=16, usePadding=True, useBatchNorm=False, usePool=True),
                Block_CNN(nbIn=16, nbOut=24, usePadding=True, useBatchNorm=True, usePool=True),
                Block_CNN(nbIn=24, nbOut=32, usePadding=False, useBatchNorm=False, usePool=False),
                Block_CNN(nbIn=32, nbOut=48, usePadding=False, useBatchNorm=False, usePool=True),
                torch.nn.Dropout(dropout), torch.nn.Flatten(), # => (48 x 2 x 2)
                torch.nn.Linear((48*2*2), 48), # reduce dim to avoid exploding MoE (... per experts)
                Block_FFD_Moe(nbIn=48, nbOut=nbClasses, nbExperts=nbExperts, topK=topK, fromCNN=False, memMode=memoryMode)])
        elif modelConfig == "medium":
            return VisionModelMoe(nbClasses=nbClasses, wImp=wImp, wLoad=wLoad, blocks=[
                Block_CNN(nbIn=3, nbOut=16, usePadding=True, useBatchNorm=False, usePool=False),
                Block_CNN(nbIn=16, nbOut=32, usePadding=True, useBatchNorm=True, usePool=True),
                Block_CNN(nbIn=32, nbOut=32, usePadding=True, useBatchNorm=False, usePool=False),
                Block_CNN(nbIn=32, nbOut=64, usePadding=True, useBatchNorm=False, usePool=False),
                Block_CNN(nbIn=64, nbOut=128, usePadding=False, useBatchNorm=False, usePool=True),
                Block_CNN(nbIn=128, nbOut=192, usePadding=False, useBatchNorm=True, usePool=True),
                torch.nn.Dropout(dropout), torch.nn.Flatten(), # => (198 x 2 x 2)
                torch.nn.Linear((192*2*2), 128), # reduce dim to avoid exploding MoE (... per experts)
                Block_FFD_Moe(nbIn=128, nbOut=nbClasses, nbExperts=nbExperts, topK=topK, fromCNN=False, memMode=memoryMode)])
        elif modelConfig == "large":
            return VisionModelMoe(nbClasses=nbClasses, wImp=wImp, wLoad=wLoad, blocks=[
                Block_CNN(nbIn=3, nbOut=64, usePadding=True, useBatchNorm=False, usePool=False),
                Block_CNN(nbIn=64, nbOut=128, usePadding=True, useBatchNorm=True, usePool=True),
                Block_CNN(nbIn=128, nbOut=128, usePadding=True, useBatchNorm=False, usePool=False),
                Block_CNN(nbIn=128, nbOut=128, usePadding=True, useBatchNorm=False, usePool=False),
                Block_CNN(nbIn=128, nbOut=256, usePadding=False, useBatchNorm=False, usePool=True),
                Block_CNN(nbIn=256, nbOut=512, usePadding=False, useBatchNorm=True, usePool=True),
                torch.nn.Dropout(dropout), torch.nn.Flatten(), # => (512 x 2 x 2)
                torch.nn.Linear((512*2*2), 192), # reduce dim to avoid exploding MoE (... per experts)
                Block_FFD_Moe(nbIn=192, nbOut=nbClasses, nbExperts=nbExperts, topK=topK, fromCNN=False, memMode=memoryMode)])
        else:
            raise ValueError(f"unknown {modelConfig = }")
    
    @staticmethod
    def get_cifar_v2(
            nbClasses:int, nbExperts:int, topK:int,
            modelConfig:typing.Literal["small", "medium", "large"],
            wImp:float, wLoad:float, memoryMode:bool, dropout=0.25)->"VisionModelMoe":
        """the V2 will use MOE rigth before flattening"""
        if modelConfig == "small":
            return VisionModelMoe(nbClasses=nbClasses, wImp=wImp, wLoad=wLoad, blocks=[
                Block_CNN(nbIn=3, nbOut=16, usePadding=True, useBatchNorm=False, usePool=True),
                Block_CNN(nbIn=16, nbOut=24, usePadding=True, useBatchNorm=True, usePool=True),
                Block_CNN(nbIn=24, nbOut=32, usePadding=False, useBatchNorm=False, usePool=False),
                Block_CNN(nbIn=32, nbOut=48, usePadding=False, useBatchNorm=False, usePool=True),
                Block_FFD_Moe(nbIn=48, nbOut=48, nbExperts=nbExperts, topK=topK, fromCNN=True, memMode=memoryMode),
                torch.nn.Dropout(dropout), torch.nn.Flatten(), # => (48 x 2 x 2)
                torch.nn.Linear((48*2*2), nbClasses)])
        elif modelConfig == "medium":
            return VisionModelMoe(nbClasses=nbClasses, wImp=wImp, wLoad=wLoad, blocks=[
                Block_CNN(nbIn=3, nbOut=16, usePadding=True, useBatchNorm=False, usePool=False),
                Block_CNN(nbIn=16, nbOut=32, usePadding=True, useBatchNorm=True, usePool=True),
                Block_CNN(nbIn=32, nbOut=32, usePadding=True, useBatchNorm=False, usePool=False),
                Block_CNN(nbIn=32, nbOut=64, usePadding=True, useBatchNorm=False, usePool=False),
                Block_CNN(nbIn=64, nbOut=128, usePadding=False, useBatchNorm=False, usePool=True),
                Block_CNN(nbIn=128, nbOut=192, usePadding=False, useBatchNorm=True, usePool=True),
                Block_FFD_Moe(nbIn=192, nbOut=192, nbExperts=nbExperts, topK=topK, fromCNN=True, memMode=memoryMode),
                torch.nn.Dropout(dropout), torch.nn.Flatten(), # => (198 x 2 x 2)
                torch.nn.Linear((192*2*2), nbClasses)])
        elif modelConfig == "large":
            return VisionModelMoe(nbClasses=nbClasses, wImp=wImp, wLoad=wLoad, blocks=[
                Block_CNN(nbIn=3, nbOut=64, usePadding=True, useBatchNorm=False, usePool=False),
                Block_CNN(nbIn=64, nbOut=128, usePadding=True, useBatchNorm=True, usePool=True),
                Block_CNN(nbIn=128, nbOut=128, usePadding=True, useBatchNorm=False, usePool=False),
                Block_CNN(nbIn=128, nbOut=128, usePadding=True, useBatchNorm=False, usePool=False),
                Block_CNN(nbIn=128, nbOut=256, usePadding=False, useBatchNorm=False, usePool=True),
                Block_CNN(nbIn=256, nbOut=256, usePadding=False, useBatchNorm=True, usePool=True),
                Block_FFD_Moe(nbIn=256, nbOut=256, nbExperts=nbExperts, topK=topK, fromCNN=True, memMode=memoryMode),
                torch.nn.Dropout(dropout), torch.nn.Flatten(), # => (256 x 2 x 2)
                torch.nn.Linear((256*2*2), nbClasses),])
        else:
            raise ValueError(f"unknown {modelConfig = }")
        
    @staticmethod
    def get_cifar_v3(
            nbClasses:int, nbExperts:int, topK:int,
            modelConfig:typing.Literal["small", "medium", "large"],
            wImp:float, wLoad:float, memoryMode:bool, dropout=0.25)->"VisionModelMoe":
        """the V3 will use MOE before the last CNN"""
        if modelConfig == "small":
            return VisionModelMoe(nbClasses=nbClasses, wImp=wImp, wLoad=wLoad, blocks=[
                Block_CNN(nbIn=3, nbOut=16, usePadding=True, useBatchNorm=False, usePool=True),
                Block_CNN(nbIn=16, nbOut=24, usePadding=True, useBatchNorm=True, usePool=True),
                Block_CNN(nbIn=24, nbOut=32, usePadding=False, useBatchNorm=False, usePool=False),
                Block_FFD_Moe(nbIn=32, nbOut=32, nbExperts=nbExperts, topK=topK, fromCNN=True, memMode=memoryMode),
                Block_CNN(nbIn=32, nbOut=48, usePadding=False, useBatchNorm=False, usePool=True),
                torch.nn.Dropout(dropout), torch.nn.Flatten(), # => (48 x 2 x 2)
                torch.nn.Linear((48*2*2), nbClasses)])
        elif modelConfig == "medium":
            return VisionModelMoe(nbClasses=nbClasses, wImp=wImp, wLoad=wLoad, blocks=[
                Block_CNN(nbIn=3, nbOut=16, usePadding=True, useBatchNorm=False, usePool=False),
                Block_CNN(nbIn=16, nbOut=32, usePadding=True, useBatchNorm=True, usePool=True),
                Block_CNN(nbIn=32, nbOut=32, usePadding=True, useBatchNorm=False, usePool=False),
                Block_CNN(nbIn=32, nbOut=64, usePadding=True, useBatchNorm=False, usePool=False),
                Block_CNN(nbIn=64, nbOut=128, usePadding=False, useBatchNorm=False, usePool=True),
                Block_FFD_Moe(nbIn=128, nbOut=128, nbExperts=nbExperts, topK=topK, fromCNN=True, memMode=memoryMode),
                Block_CNN(nbIn=128, nbOut=192, usePadding=False, useBatchNorm=True, usePool=True),
                torch.nn.Dropout(dropout), torch.nn.Flatten(), # => (198 x 2 x 2)
                torch.nn.Linear((192*2*2), nbClasses)])
        elif modelConfig == "large":
            return VisionModelMoe(nbClasses=nbClasses, wImp=wImp, wLoad=wLoad, blocks=[
                Block_CNN(nbIn=3, nbOut=64, usePadding=True, useBatchNorm=False, usePool=False),
                Block_CNN(nbIn=64, nbOut=128, usePadding=True, useBatchNorm=True, usePool=True),
                Block_CNN(nbIn=128, nbOut=128, usePadding=True, useBatchNorm=False, usePool=False),
                Block_CNN(nbIn=128, nbOut=128, usePadding=True, useBatchNorm=False, usePool=False),
                Block_CNN(nbIn=128, nbOut=256, usePadding=False, useBatchNorm=False, usePool=True),
                Block_FFD_Moe(nbIn=256, nbOut=256, nbExperts=nbExperts, topK=topK, fromCNN=True, memMode=memoryMode),
                Block_CNN(nbIn=256, nbOut=256, usePadding=False, useBatchNorm=True, usePool=True),
                torch.nn.Dropout(dropout), torch.nn.Flatten(), # => (256 x 2 x 2)
                torch.nn.Linear((256*2*2), nbClasses),])
        else:
            raise ValueError(f"unknown {modelConfig = }")
    
    @staticmethod
    def get_cifar_v4(
            nbClasses:int, nbExperts:int, topK:int,
            modelConfig:typing.Literal["small", "medium", "large"],
            wImp:float, wLoad:float, memoryMode:bool, dropout=0.25)->"VisionModelMoe":
        """the V4 will use multiple MOE: before and rigth after the 2nd last CNN"""
        if modelConfig == "small":
            return VisionModelMoe(nbClasses=nbClasses, wImp=wImp, wLoad=wLoad, blocks=[
                Block_CNN(nbIn=3, nbOut=16, usePadding=True, useBatchNorm=False, usePool=True),
                Block_CNN(nbIn=16, nbOut=24, usePadding=True, useBatchNorm=True, usePool=True),
                Block_FFD_Moe(nbIn=24, nbOut=24, nbExperts=nbExperts, topK=topK, fromCNN=True, memMode=memoryMode),
                Block_CNN(nbIn=24, nbOut=32, usePadding=False, useBatchNorm=False, usePool=False),
                Block_FFD_Moe(nbIn=32, nbOut=32, nbExperts=nbExperts, topK=topK, fromCNN=True, memMode=memoryMode),
                Block_CNN(nbIn=32, nbOut=48, usePadding=False, useBatchNorm=False, usePool=True),
                torch.nn.Dropout(dropout), torch.nn.Flatten(), # => (48 x 2 x 2)
                torch.nn.Linear((48*2*2), nbClasses)])
        elif modelConfig == "medium":
            return VisionModelMoe(nbClasses=nbClasses, wImp=wImp, wLoad=wLoad, blocks=[
                Block_CNN(nbIn=3, nbOut=16, usePadding=True, useBatchNorm=False, usePool=False),
                Block_CNN(nbIn=16, nbOut=32, usePadding=True, useBatchNorm=True, usePool=True),
                Block_CNN(nbIn=32, nbOut=32, usePadding=True, useBatchNorm=False, usePool=False),
                Block_CNN(nbIn=32, nbOut=64, usePadding=True, useBatchNorm=False, usePool=False),
                Block_FFD_Moe(nbIn=64, nbOut=64, nbExperts=nbExperts, topK=topK, fromCNN=True, memMode=memoryMode),
                Block_CNN(nbIn=64, nbOut=128, usePadding=False, useBatchNorm=False, usePool=True),
                Block_FFD_Moe(nbIn=128, nbOut=128, nbExperts=nbExperts, topK=topK, fromCNN=True, memMode=memoryMode),
                Block_CNN(nbIn=128, nbOut=192, usePadding=False, useBatchNorm=True, usePool=True),
                torch.nn.Dropout(dropout), torch.nn.Flatten(), # => (198 x 2 x 2)
                torch.nn.Linear((192*2*2), nbClasses)])
        elif modelConfig == "large":
            return VisionModelMoe(nbClasses=nbClasses, wImp=wImp, wLoad=wLoad, blocks=[
                Block_CNN(nbIn=3, nbOut=64, usePadding=True, useBatchNorm=False, usePool=False),
                Block_CNN(nbIn=64, nbOut=128, usePadding=True, useBatchNorm=True, usePool=True),
                Block_CNN(nbIn=128, nbOut=128, usePadding=True, useBatchNorm=False, usePool=False),
                Block_CNN(nbIn=128, nbOut=128, usePadding=True, useBatchNorm=False, usePool=False),
                Block_FFD_Moe(nbIn=128, nbOut=128, nbExperts=nbExperts, topK=topK, fromCNN=True, memMode=memoryMode),
                Block_CNN(nbIn=128, nbOut=256, usePadding=False, useBatchNorm=False, usePool=True),
                Block_FFD_Moe(nbIn=256, nbOut=256, nbExperts=nbExperts, topK=topK, fromCNN=True, memMode=memoryMode),
                Block_CNN(nbIn=256, nbOut=256, usePadding=False, useBatchNorm=True, usePool=True),
                torch.nn.Dropout(dropout), torch.nn.Flatten(), # => (256 x 2 x 2)
                torch.nn.Linear((256*2*2), nbClasses),])
        else:
            raise ValueError(f"unknown {modelConfig = }")
    
    @staticmethod
    def get_cifar_v5(
            nbClasses:int, nbExperts:int, topK:int,
            modelConfig:typing.Literal["small", "medium", "large"],
            wImp:float, wLoad:float, memoryMode:bool, dropout=0.25)->"VisionModelMoe":
        """the V5 will use multiple MOE: before the 2nd last CNN"""
        if modelConfig == "small":
            return VisionModelMoe(nbClasses=nbClasses, wImp=wImp, wLoad=wLoad, blocks=[
                Block_CNN(nbIn=3, nbOut=16, usePadding=True, useBatchNorm=False, usePool=True),
                Block_CNN(nbIn=16, nbOut=24, usePadding=True, useBatchNorm=True, usePool=True),
                Block_FFD_Moe(nbIn=24, nbOut=24, nbExperts=nbExperts, topK=topK, fromCNN=True, memMode=memoryMode),
                Block_CNN(nbIn=24, nbOut=32, usePadding=False, useBatchNorm=False, usePool=False),
                Block_CNN(nbIn=32, nbOut=48, usePadding=False, useBatchNorm=False, usePool=True),
                torch.nn.Dropout(dropout), torch.nn.Flatten(), # => (48 x 2 x 2)
                torch.nn.Linear((48*2*2), nbClasses)])
        elif modelConfig == "medium":
            return VisionModelMoe(nbClasses=nbClasses, wImp=wImp, wLoad=wLoad, blocks=[
                Block_CNN(nbIn=3, nbOut=16, usePadding=True, useBatchNorm=False, usePool=False),
                Block_CNN(nbIn=16, nbOut=32, usePadding=True, useBatchNorm=True, usePool=True),
                Block_CNN(nbIn=32, nbOut=32, usePadding=True, useBatchNorm=False, usePool=False),
                Block_CNN(nbIn=32, nbOut=64, usePadding=True, useBatchNorm=False, usePool=False),
                Block_FFD_Moe(nbIn=64, nbOut=64, nbExperts=nbExperts, topK=topK, fromCNN=True, memMode=memoryMode),
                Block_CNN(nbIn=64, nbOut=128, usePadding=False, useBatchNorm=False, usePool=True),
                Block_CNN(nbIn=128, nbOut=192, usePadding=False, useBatchNorm=True, usePool=True),
                torch.nn.Dropout(dropout), torch.nn.Flatten(), # => (198 x 2 x 2)
                torch.nn.Linear((192*2*2), nbClasses)])
        elif modelConfig == "large":
            return VisionModelMoe(nbClasses=nbClasses, wImp=wImp, wLoad=wLoad, blocks=[
                Block_CNN(nbIn=3, nbOut=64, usePadding=True, useBatchNorm=False, usePool=False),
                Block_CNN(nbIn=64, nbOut=128, usePadding=True, useBatchNorm=True, usePool=True),
                Block_CNN(nbIn=128, nbOut=128, usePadding=True, useBatchNorm=False, usePool=False),
                Block_CNN(nbIn=128, nbOut=128, usePadding=True, useBatchNorm=False, usePool=False),
                Block_FFD_Moe(nbIn=128, nbOut=128, nbExperts=nbExperts, topK=topK, fromCNN=True, memMode=memoryMode),
                Block_CNN(nbIn=128, nbOut=256, usePadding=False, useBatchNorm=False, usePool=True),
                Block_CNN(nbIn=256, nbOut=256, usePadding=False, useBatchNorm=True, usePool=True),
                torch.nn.Dropout(dropout), torch.nn.Flatten(), # => (256 x 2 x 2)
                torch.nn.Linear((256*2*2), nbClasses),])
        else:
            raise ValueError(f"unknown {modelConfig = }")
    
    @staticmethod
    def get_mnist_v1(
            nbClasses:int, nbExperts:int, topK:int,
            modelConfig:typing.Literal["small", "medium", "large"],
            wImp:float, wLoad:float, dropout=0.25)->"VisionModelMoe":
        """the V1 will use MOE at the end of the network to output the logits"""
        if modelConfig == "small":
            return VisionModelMoe(nbClasses=nbClasses, wImp=wImp, wLoad=wLoad, blocks=[
                Block_CNN(1, 8, usePadding=True, useBatchNorm=False, usePool=True),
                Block_CNN(8, 16, usePadding=True, useBatchNorm=True, usePool=True),
                Block_CNN(16, 24, usePadding=True, useBatchNorm=False, usePool=False),
                Block_CNN(24, 32, usePadding=False, useBatchNorm=False, usePool=True),
                torch.nn.Dropout(dropout), torch.nn.Flatten(), # => (32 x 2 x 2)
                torch.nn.Linear((32*2*2), 32), # reduce dim to avoid exploding MoE (... per experts)
                Block_FFD_Moe(nbIn=32, nbOut=nbClasses, nbExperts=nbExperts, topK=topK, fromCNN=False)])
        elif modelConfig == "medium":
            return VisionModelMoe(nbClasses=nbClasses, wImp=wImp, wLoad=wLoad, blocks=[
                Block_CNN(1, 16, usePadding=True, useBatchNorm=False, usePool=True), 
                Block_CNN(16, 24, usePadding=True, useBatchNorm=True, usePool=True),
                Block_CNN(24, 32, usePadding=True, useBatchNorm=False, usePool=False),
                Block_CNN(32, 48, usePadding=False, useBatchNorm=False, usePool=True),
                torch.nn.Dropout(dropout), torch.nn.Flatten(), # => (198 x 2 x 2)
                torch.nn.Linear((48*2*2), 48), # reduce dim to avoid exploding MoE (... per experts)
                Block_FFD_Moe(nbIn=48, nbOut=nbClasses, nbExperts=nbExperts, topK=topK, fromCNN=False)])
        elif modelConfig == "large":
            return VisionModelMoe(nbClasses=nbClasses, wImp=wImp, wLoad=wLoad, blocks=[
                Block_CNN(1, 16, usePadding=True, useBatchNorm=False, usePool=False),
                Block_CNN(16, 32, usePadding=True, useBatchNorm=True, usePool=True),
                Block_CNN(32, 32, usePadding=True, useBatchNorm=False, usePool=False),
                Block_CNN(32, 64, usePadding=True, useBatchNorm=False, usePool=False),
                Block_CNN(64, 128, usePadding=True, useBatchNorm=False, usePool=True),
                Block_CNN(128, 128, usePadding=False, useBatchNorm=True, usePool=True),
                torch.nn.Dropout(dropout), torch.nn.Flatten(), # => (512 x 2 x 2)
                torch.nn.Linear((128*2*2), 128), # reduce dim to avoid exploding MoE (... per experts)
                Block_FFD_Moe(nbIn=128, nbOut=nbClasses, nbExperts=nbExperts, topK=topK, fromCNN=False)])
        else:
            raise ValueError(f"unknown {modelConfig = }")