import torch
import numpy
import typing

from prettytable import PrettyTable



################# utils #################

def countLayersparameters(model: torch.nn.Module):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        table.add_row([name, f"{params:_d}"])
        total_params += params
    print(table)
    print(f"Total Trainable Params: {total_params:_d}")
    return total_params

def countTotalParameters(iter:typing.Iterator[torch.nn.Parameter])->int:
        return sum(int(numpy.prod(params.size())) for params in iter)




################# Models #################

class BasicImageClassifModel(torch.nn.Module):
    def __init__(self, layers:list[tuple[int, bool, bool, bool]],
                 nbChanels:int, nbClasses:int, 
                 botleneckShape:tuple[int, int], **kwargs) -> None:
        """layers: list[(nbFeatureOut, usePadding, useBatchNorm, usePool)]"""
        super().__init__(**kwargs)
        self.nbChanels: int = nbChanels
        self.nbClasses: int = nbClasses
        self.botleneckShape: tuple[int, int] = botleneckShape
        # create the network
        self.layers: list[torch.nn.Module] = []
        nbIn: int = nbChanels
        for i, (nbOut, usePadd, useNorm, usePool) in enumerate(layers):
            block = torch.nn.Sequential(
                torch.nn.Conv2d(
                    nbIn, nbOut, kernel_size=3, bias=False, 
                    padding=("same" if usePadd else 0)), 
                *[nn for nn in [torch.nn.BatchNorm2d(nbOut)] if useNorm], 
                torch.nn.ReLU(), 
                *[nn for nn in [torch.nn.MaxPool2d(kernel_size=2)] if usePool])
            self.add_module(f"CNN_block{i+1}", block)
            self.layers.append(block)
            nbIn = nbOut
        nbIn = (self.botleneckShape[0] * self.botleneckShape[1] * layers[-1][0])
        outputBlock =  torch.nn.Sequential(
            torch.nn.Dropout(0.25),
            torch.nn.Flatten(start_dim=1, end_dim=-1),
            torch.nn.Linear(nbIn, self.nbClasses))
        self.add_module(f"outputBlock", outputBlock)
        self.layers.append(outputBlock)
    
    def forward(self, x:torch.Tensor)->torch.Tensor:
        if len(x.shape) == 3:
            x = x[None, ...]
        #print(x.shape)
        for layer in self.layers:
            try:
                x = layer(x)
            except Exception as err:
                print(x.shape, layer)
                raise err
        return x
    
    @staticmethod
    def get_MNIST_like_28x28x1(
            device:torch.device, nbClasses:int,
            modelConfig:typing.Literal["small", "medium", "large"], **kwargs):
        if modelConfig == "large":
            model = BasicImageClassifModel(
                layers=[(16, True, False, False), (32, True, True, True),
                        (32, True, False, False), (64, True, False, False), 
                        (128, True, False, True), (128, True, True, True), ],
                nbChanels=1, nbClasses=nbClasses, botleneckShape=(3,3), **kwargs)
        elif modelConfig == "medium":
            model = BasicImageClassifModel(
                layers=[(16, True, False, True), (24, True, True, True),
                        (32, True, False, False), (48, True, False, True), ],
                nbChanels=1, nbClasses=nbClasses, botleneckShape=(3,3), **kwargs)
        elif modelConfig == "small":
            model = BasicImageClassifModel(
                layers=[(8, True, False, True), (16, True, True, True),
                        (24, True, False, False), (32, True, False, True), ],
                nbChanels=1, nbClasses=nbClasses, botleneckShape=(3,3), **kwargs)
        else: 
            raise ValueError(f"unknown {modelConfig = }")
        model.to(device)
        optim = torch.optim.AdamW(model.parameters(), lr=0.001)
        loss = torch.nn.CrossEntropyLoss().to(device)
        return (model, optim, loss)
    
    @staticmethod
    def get_Cifar_like_32x32x3(
            device:torch.device, nbClasses:int,
            modelConfig:typing.Literal["small", "medium", "large"], **kwargs):
        if modelConfig == "large":
            model = BasicImageClassifModel(
                layers=[(64, True, False, False), (128, True, True, True),
                        (128, True, False, False), (128, True, False, False), 
                        (256, True, False, True), (512, True, True, True), ],
                nbChanels=3, nbClasses=nbClasses, botleneckShape=(4,4), **kwargs)
        elif modelConfig == "medium":
            model = BasicImageClassifModel(
                layers=[(16, True, False, False), (32, True, True, True),
                        (32, True, False, False), (64, True, False, False), 
                        (128, True, False, True), (256, True, True, True), ],
                nbChanels=3, nbClasses=nbClasses, botleneckShape=(4,4), **kwargs)
        elif modelConfig == "small":
            model = BasicImageClassifModel(
                layers=[(16, True, False, True), (24, True, True, True),
                        (32, True, False, False), (48, True, False, True), ],
                nbChanels=3, nbClasses=nbClasses, botleneckShape=(4,4), **kwargs)
        else: 
            raise ValueError(f"unknown {modelConfig = }")
        model.to(device)
        optim = torch.optim.AdamW(model.parameters(), lr=0.001)
        loss = torch.nn.CrossEntropyLoss().to(device)
        return (model, optim, loss)
    
    @staticmethod
    def get_Cifar_like_32x32x3_moe(
            device:torch.device, nbClasses:int, nbExperts:int,
            expertsModelConfig:typing.Literal["small", "medium", "large"],
            gatingModelConfig:typing.Literal["small", "medium", "large"]):
        allExperts: list[BasicImageClassifModel] = [
            BasicImageClassifModel.get_Cifar_like_32x32x3(
                device=device, nbClasses=nbClasses, modelConfig=expertsModelConfig)[0]
            for _ in range(nbExperts)]
        gatingModel, _, loss = BasicImageClassifModel.get_Cifar_like_32x32x3(
            device=device, nbClasses=nbExperts, modelConfig=gatingModelConfig)
        return (gatingModel, allExperts, loss)
    