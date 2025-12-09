import torch
from torch.utils.data import DataLoader
import attrs
import typing

from handleDatas import CustomLoader, HandleClassifDatas

from holo.profilers import ProgressBar
from holo.prettyFormats import SingleLinePrinter



@attrs.frozen
class ResultsClassif():
    loss: float
    accuracy: float

    def __str__(self) -> str:
        return f"(loss: {self.loss:.4g}, accuracy: {self.accuracy:.2%})"


class HistoryClassification(list[tuple[ResultsClassif, ResultsClassif]]):
    pass


def train_model_classif(
        *, model, optimizer, criterion, device: torch.device,
        datasHandler: HandleClassifDatas,
        nbEpoches: int, history: HistoryClassification | None = None) -> ResultsClassif:
    return train_model_classif_base(
        model=model, optimizer=optimizer, criterion=criterion, device=device,
        datasTrain=datasHandler.train_cLoader(), datasTest=datasHandler.test_cLoader(),
        nbEpoches=nbEpoches, history=history)

def train_model_classif_base(
        *, model, optimizer, criterion, device: torch.device,
        datasTrain: CustomLoader, datasTest:CustomLoader,
        nbEpoches: int, history: HistoryClassification | None = None) -> ResultsClassif:
    if history is None:
        history = HistoryClassification()
    assert nbEpoches > 0
    for epoch in range(nbEpoches):
        running_loss = 0.0
        running_accuracy = 0.0
        nbDone: int = 0
        pbar = ProgressBar.simpleConfig(
            nbSteps=len(datasTrain), taskName="train batches", useEma=True)
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
        testResults = eval_model_classif(
            model=model, criterion=criterion, device=device, 
            datas=datasTest, verbose=False)
        meanLoss = (running_loss / len(datasTest))
        meanAccuracy = (running_accuracy / nbDone)
        history.append((ResultsClassif(loss=meanLoss, accuracy=meanAccuracy), testResults))
        print(f'Epoch {epoch+1}, train: {history[-1][0]}, test: {testResults}')
    return history[-1][0]


def eval_model_classif(
        *, model, criterion, device: torch.device,
        datas: CustomLoader, verbose: bool) -> ResultsClassif:
    running_loss = 0.0
    running_accuracy = 0.0
    nbDone: int = 0
    pbar = ProgressBar.simpleConfig(
        nbSteps=len(datas), taskName="test batches", useEma=True)
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
    return ResultsClassif(loss=meanLoss, accuracy=meanAccuracy)
