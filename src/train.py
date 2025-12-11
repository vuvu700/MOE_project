import torch
import numpy
from torch.utils.data import DataLoader
import attrs

import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from handleDatas import CustomLoader, HandleClassifDatas

from holo.profilers import ProgressBar, Profiler
from holo.prettyFormats import SingleLinePrinter


@attrs.frozen
class ResultClassif():
    loss: float
    confusionMatrix: "ConfusionMatrix"
    
    def accuracy(self)->float:
        return self.confusionMatrix.accuracy()
    
    def __str__(self) -> str:
        return f"(loss: {self.loss:.4g}, accuracy: {self.accuracy():.2%})"

@attrs.frozen
class EpochResultClassif():
    epochID: int
    train: ResultClassif
    test: ResultClassif
    lr: float
    
    def __str__(self) -> str:
        return f"Epoch {self.epochID}, train: {self.train}, test: {self.test}, lr: {self.lr:.4e}"

class HistoryClassification(list[EpochResultClassif]):
    
    def plot(self):
        axs: list[Axes]
        fig, axs = plt.subplots(ncols=1, nrows=3, sharex=True)
        epoches = list(range(1, 1+len(self)))
        
        axs[0].plot(epoches, [r.train.loss for r in self], label="loss_train")
        axs[0].plot(epoches, [r.test.loss for r in self], label="loss_test")
        axs[0].legend()
        axs[0].set_yscale("log")
        axs[0].grid(True)
        
        axs[1].plot(epoches, [r.train.accuracy() for r in self], label="accuracy_train")
        axs[1].plot(epoches, [r.test.accuracy() for r in self], label="accuracy_test")
        axs[1].legend()
        axs[1].grid(True)
        
        axs[2].plot(epoches, [r.lr for r in self], label="lr")
        axs[2].legend()
        axs[2].set_yscale("log")
        axs[2].grid(True)
        
        return fig


class ConfusionMatrix():
    def __init__(self, nbClasses:int) -> None:
        self.matrix = numpy.zeros((nbClasses, nbClasses), dtype=numpy.int32)
        """(rows:predicted, cols:truth)"""
    
    def nb_total(self)->int:
        return int(self.matrix.sum())
    def nb_expected(self, classIndex:int)->int:
        return int(self.matrix[:, classIndex].sum())
    def nb_predicted(self, classIndex:int)->int:
        return int(self.matrix[classIndex, :].sum())
    def nb_truePositive(self, classIndex:int|None)->int:
        if classIndex is not None:
            return int(self.matrix[classIndex, classIndex])
        return int(self.matrix.diagonal().sum())
            
    
    def accuracy(self)->float:
        return self.nb_truePositive(None) / self.nb_total()
    def classExpectedBalance(self, classIndex:int)->float:
        """how much the class was expected"""
        return self.nb_expected(classIndex) / self.nb_total()
    def classPredictedBalance(self, classIndex:int)->float:
        """how much the class was predicted"""
        return self.nb_predicted(classIndex) / self.matrix.sum()
    def classPrecision(self, classIndex:int)->float:
        """when predicted, how much were a good"""
        return  (self.nb_truePositive(classIndex) / self.nb_predicted(classIndex))
    def classHitRate(self, classIndex:int)->float:
        """when expected, how much were a good"""
        return  (self.nb_truePositive(classIndex) / self.nb_expected(classIndex))
    
    def worstK_confusions(self, k:int)->list[tuple[float, int, int]]:
        """return the list of the k worst confusions:
        list of tuple(nbPredictions/totalErrors, truth, predicted)"""
        totalErrs = self.nb_total() - self.nb_truePositive(None)
        return sorted(
            [(float(self.matrix[clPred, clTrue]/totalErrs), clTrue, clPred)
             for clPred in range(self.matrix.shape[0]) for clTrue in range(self.matrix.shape[0])
             if clPred != clTrue],
            reverse=True)[: k]
    
    def step(self, predLabels:list[int], truthLabels:list[int])->None:
        for pred, truth in zip(predLabels, truthLabels):
            self.matrix[pred, truth] += 1



def train_model_classif(
        *, model:torch.nn.Module, optimizer:torch.optim.Optimizer, 
        criterion:torch.nn.Module, device: torch.device,
        datasHandler: HandleClassifDatas,
        nbEpoches: int, history: HistoryClassification | None = None) -> ResultClassif:
    return train_model_classif_base(
        model=model, optimizer=optimizer, criterion=criterion, device=device,
        datasTrain=datasHandler.train_cLoader(), datasTest=datasHandler.test_cLoader(),
        nbClasses=datasHandler.nbClasses, nbEpoches=nbEpoches, history=history)

def train_model_classif_base(
        *, model:torch.nn.Module, optimizer:torch.optim.Optimizer, 
        criterion:torch.nn.Module, device: torch.device,
        datasTrain: CustomLoader, datasTest:CustomLoader, nbClasses:int,
        nbEpoches: int, history: HistoryClassification | None = None) -> ResultClassif:
    if history is None:
        history = HistoryClassification()
    assert nbEpoches > 0
    startEpochID: int = (0 if len(history) == 0 else history[-1].epochID)
    for epochID in range(startEpochID+1, (startEpochID+1+nbEpoches)):
        running_loss = 0.0
        trainConfMatrix = ConfusionMatrix(nbClasses=nbClasses)
        pbar = ProgressBar.simpleConfig(
            len(datasTrain), "train batches", newLineWhenFinished=False, updateEvery=1/20)
        for inputs, labels, _ in datasTrain(device):
            model.train(True)
            optimizer.zero_grad()
            outputs: torch.Tensor = model(inputs)
            loss: torch.Tensor = criterion(outputs, labels)
            loss.backward()
            model.eval()
            optimizer.step()
            running_loss += loss.item()
            predLabels = torch.argmax(outputs.detach(), dim=-1)
            trainConfMatrix.step(predLabels=predLabels.tolist(), truthLabels=labels.tolist())
            pbar.step(1)
        testResult = eval_model_classif(
            model=model, criterion=criterion, device=device, 
            datas=datasTest, nbClasses=nbClasses, verbose=True)
        meanLoss = (running_loss / len(datasTrain))
        trainResult = ResultClassif(loss=meanLoss, confusionMatrix=trainConfMatrix)
        history.append(EpochResultClassif(
            epochID, trainResult, testResult, lr=optimizer.param_groups[0]['lr']))
        print(history[-1])
    return history[-1].train


def eval_model_classif(
        *, model:torch.nn.Module, criterion:torch.nn.Module, device:torch.device,
        datas:CustomLoader, nbClasses:int, verbose:bool) -> ResultClassif:
    running_loss = 0.0
    confMatrix = ConfusionMatrix(nbClasses=nbClasses)
    pbar = ProgressBar.simpleConfig(
        len(datas), "test batches", newLineWhenFinished=False, updateEvery=1/20)
    model.eval()
    for inputs, labels, _ in datas(device):
        with torch.no_grad():
            outputs: torch.Tensor = model(inputs)
        loss: torch.Tensor = criterion(outputs, labels)
        running_loss += loss.item()
        predLabels = torch.argmax(outputs.detach(), dim=-1)
        confMatrix.step(predLabels=predLabels.tolist(), truthLabels=labels.tolist())
        if verbose is True:
            pbar.step()
    meanLoss = (running_loss / len(datas))
    return ResultClassif(loss=meanLoss, confusionMatrix=confMatrix)
