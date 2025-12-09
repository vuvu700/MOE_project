import torch
from torch.utils.data import DataLoader
import attrs

import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from handleDatas import CustomLoader, HandleClassifDatas

from holo.profilers import ProgressBar
from holo.prettyFormats import SingleLinePrinter



@attrs.frozen
class ResultClassif():
    loss: float
    accuracy: float
    
    def __str__(self) -> str:
        return f"(loss: {self.loss:.4g}, accuracy: {self.accuracy:.2%})"

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
        
        axs[1].plot(epoches, [r.train.accuracy for r in self], label="accuracy_train")
        axs[1].plot(epoches, [r.test.accuracy for r in self], label="accuracy_test")
        axs[1].legend()
        axs[1].grid(True)
        
        axs[2].plot(epoches, [r.lr for r in self], label="lr")
        axs[2].legend()
        axs[2].set_yscale("log")
        axs[2].grid(True)
        
        return fig


def train_model_classif(
        *, model, optimizer, criterion, device: torch.device,
        datasHandler: HandleClassifDatas,
        nbEpoches: int, history: HistoryClassification | None = None) -> ResultClassif:
    return train_model_classif_base(
        model=model, optimizer=optimizer, criterion=criterion, device=device,
        datasTrain=datasHandler.train_cLoader(), datasTest=datasHandler.test_cLoader(),
        nbEpoches=nbEpoches, history=history)

def train_model_classif_base(
        *, model, optimizer, criterion, device: torch.device,
        datasTrain: CustomLoader, datasTest:CustomLoader,
        nbEpoches: int, history: HistoryClassification | None = None) -> ResultClassif:
    if history is None:
        history = HistoryClassification()
    assert nbEpoches > 0
    startEpochID: int = (0 if len(history) == 0 else history[-1].epochID)
    for epochID in range(startEpochID+1, (startEpochID+1+nbEpoches)):
        running_loss = 0.0
        running_accuracy = 0.0
        nbDone: int = 0
        pbar = ProgressBar.simpleConfig(
            len(datasTrain), "train batches", newLineWhenFinished=False, updateEvery=0.2)
        for stepIndex, (inputs, labels, _) in enumerate(datasTrain(device)):
            model.train(True)
            optimizer.zero_grad()
            outputs: torch.Tensor = model(inputs)
            loss: torch.Tensor = criterion(outputs, labels)
            loss.backward()
            model.eval()
            optimizer.step()
            running_loss += loss.item()
            running_accuracy += torch.sum(labels == torch.argmax(outputs.detach(), dim=-1)).item()
            nbDone += inputs.size(dim=0)
            pbar.step(1)
        testResult = eval_model_classif(
            model=model, criterion=criterion, device=device, 
            datas=datasTest, verbose=False)
        meanLoss = (running_loss / len(datasTrain))
        meanAccuracy = (running_accuracy / nbDone)
        trainResult = ResultClassif(loss=meanLoss, accuracy=meanAccuracy)
        history.append(EpochResultClassif(
            epochID, trainResult, testResult, lr=optimizer.param_groups[0]['lr']))
        print(history[-1])
    return history[-1].train


def eval_model_classif(
        *, model, criterion, device: torch.device,
        datas: CustomLoader, verbose: bool) -> ResultClassif:
    running_loss = 0.0
    running_accuracy = 0.0
    nbDone: int = 0
    pbar = ProgressBar.simpleConfig(
        len(datas), "test batches", newLineWhenFinished=True, updateEvery=0.2)
    model.eval()
    for stepIndex, (inputs, labels, _) in enumerate(datas(device)):
        with torch.no_grad():
            outputs: torch.Tensor = model(inputs)
        loss: torch.Tensor = criterion(outputs, labels)
        running_loss += loss.item()
        running_accuracy += torch.sum(labels == torch.argmax(outputs.detach(), dim=-1)).item()
        nbDone += inputs.size(dim=0)
        if verbose is True:
            pbar.step()
    meanLoss = (running_loss / len(datas))
    meanAccuracy = (running_accuracy / nbDone)
    return ResultClassif(loss=meanLoss, accuracy=meanAccuracy)
