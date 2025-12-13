import torch
import numpy
from torch.utils.data import DataLoader
import attrs

import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from handleDatas import CustomLoader, HandleClassifDatas
import MoE_models.paper1

from holo.profilers import ProgressBar, Profiler
from holo.prettyFormats import prettyPrint, prettyTime


class MoeExpertsInsigths():
    ...

@attrs.define
class Moe1ExpertsInsigths(MoeExpertsInsigths):
    sumExpertsGate: numpy.ndarray[tuple[int, ], numpy.dtype[numpy.float64]]
    """(row: experts) -> mean gate of the experts overall"""
    sumPredClassesExpertsGate: numpy.ndarray[tuple[int, int, ], numpy.dtype[numpy.float64]]
    """(row: class, col: experts) -> mean gate of the experts when this was predicted"""
    nbPred: numpy.ndarray[tuple[int, ], numpy.dtype[numpy.int64]]
    """(row: class) -> nb of time the class was prdicted"""
    sumTrueClassesExpertsGate: numpy.ndarray[tuple[int, int, ], numpy.dtype[numpy.float64]]
    """(row: class, col: experts) -> mean gate of the experts when this was the label"""
    nbTruth: numpy.ndarray[tuple[int, ], numpy.dtype[numpy.int64]]
    """(row: class) -> nb of time the class was the label"""
    
    def meanExpertsGate(self):
        return (self.sumExpertsGate / float(self.nbPred.sum()))
    def meanPredClassesExpertsGate(self):
        return (self.sumPredClassesExpertsGate / self.nbPred[..., None])
    def meanTruthClassesExpertsGate(self):
        return (self.sumTrueClassesExpertsGate / self.nbPred[..., None])
    
    
    @staticmethod
    def empty(nbExperts:int, nbClasses:int)->"Moe1ExpertsInsigths":
        return Moe1ExpertsInsigths(
            sumExpertsGate=numpy.zeros((nbExperts, )),
            sumPredClassesExpertsGate=numpy.zeros((nbClasses, nbExperts, )),
            sumTrueClassesExpertsGate=numpy.zeros((nbClasses, nbExperts, )),
            nbPred=numpy.zeros((nbClasses, ), dtype=numpy.int64), 
            nbTruth=numpy.zeros((nbClasses, ), dtype=numpy.int64))

@attrs.frozen
class ResultClassif():
    loss: float
    confusionMatrix: "ConfusionMatrix"
    moeExpertsInsigths: "MoeExpertsInsigths|None"
    
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


class TrainerClassif():
    
    def __init__(
            self, model:torch.nn.Module, optimizer:torch.optim.Optimizer,
            criterion:torch.nn.Module, device:torch.device) -> None:
        self.model: torch.nn.Module = model
        self.optimizer: torch.optim.Optimizer = optimizer
        self.criterion: torch.nn.Module = criterion
        self.device: torch.device = device
        self.updateEvery: float = (1/20)
        self.history = HistoryClassification()
    
    def train_model_classif(
            self, *, datasHandler:HandleClassifDatas, nbEpoches:int,
            history:HistoryClassification|None = None)->ResultClassif:
        return self.train_model_classif_base(
            datasTrain=datasHandler.train_cLoader(), 
            datasTest=datasHandler.test_cLoader(),
            nbClasses=datasHandler.nbClasses, 
            nbEpoches=nbEpoches)

    def train_model_classif_base(
            self, *, datasTrain: CustomLoader, datasTest:CustomLoader, 
            nbClasses:int, nbEpoches:int)->ResultClassif:
        assert nbEpoches > 0
        _prof = Profiler([
            "all", "getBatch", "predict+loss", "backward", "step", 
            "metrics_base", "metrics_moe", "evaluate", "progressBar", ])
        startEpochID: int = (0 if len(self.history) == 0 else self.history[-1].epochID)
        with _prof.mesure("all"):
            for epochID in range(startEpochID+1, (startEpochID+1+nbEpoches)):
                running_loss = 0.0
                trainConfMatrix = ConfusionMatrix(nbClasses=nbClasses)
                moeExpertsInsigths: "MoeExpertsInsigths|None" = None
                pbar = self._getPBar(len(datasTrain), "train batches")
                self.model.train()
                trainDatasIterator = iter(datasTrain(self.device))
                while True:
                    try:
                        with _prof.mesure("getBatch"):
                            inputs, labels, _ = next(trainDatasIterator)
                    except StopIteration: break
                    self.optimizer.zero_grad()
                    with _prof.mesure("predict+loss"):
                        outputs, loss = self.forwardAndLoss(inputs, labels)
                    with _prof.mesure("backward"):
                        loss.backward()
                    with _prof.mesure("step"):
                        self.optimizer.step()
                    with _prof.mesure("metrics_base"):
                        running_loss += loss.item()
                        predLabels = torch.argmax(outputs.detach(), dim=-1)
                        trainConfMatrix.step(
                            predLabels=predLabels.tolist(), truthLabels=labels.tolist())
                    with _prof.mesure("metrics_moe"):
                        moeExpertsInsigths = self.computeMoeMetrics(
                            moeExpertsInsigths, outputs, labels)
                    with _prof.mesure("progressBar"):
                        pbar.step(1)
                self.model.eval()
                with _prof.mesure("evaluate"):
                    testResult = self.eval_model_classif(
                        datas=datasTest, nbClasses=nbClasses, verbose=True)
                meanLoss = (running_loss / len(datasTrain))
                trainResult = ResultClassif(
                    loss=meanLoss, confusionMatrix=trainConfMatrix, 
                    moeExpertsInsigths=moeExpertsInsigths)
                self.history.append(EpochResultClassif(
                    epochID, trainResult, testResult,
                    lr=self.optimizer.param_groups[0]['lr']))
                print(self.history[-1])
        # show the time it took
        tt = _prof.totalMesure("all")
        times = _prof.totalTimes()
        times["other"] = tt - (sum(times.values()) - tt) # type: ignore
        prettyPrint(times, specificFormats={float: lambda x: f"{x/tt:.2%}"})
        return self.history[-1].train
    
    def eval_model_classif(
            self, *, datas:CustomLoader, nbClasses:int, 
            verbose:bool) -> ResultClassif:
        running_loss = 0.0
        confMatrix = ConfusionMatrix(nbClasses=nbClasses)
        moeExpertsInsigths: "MoeExpertsInsigths|None" = None
        pbar = self._getPBar(len(datas), "test batches")
        self.model.eval()
        for inputs, labels, _ in datas(self.device):
            with torch.no_grad():
                outputs, loss = self.forwardAndLoss(inputs, labels)
            running_loss += loss.item()
            predLabels = torch.argmax(outputs.detach(), dim=-1)
            confMatrix.step(predLabels=predLabels.tolist(), truthLabels=labels.tolist())
            moeExpertsInsigths = self.computeMoeMetrics(
                moeExpertsInsigths, outputs, labels)
            if verbose is True:
                pbar.step()
        meanLoss = (running_loss / len(datas))
        return ResultClassif(
            loss=meanLoss, confusionMatrix=confMatrix,
            moeExpertsInsigths=moeExpertsInsigths)
    
    def forwardAndLoss(
            self, inputs:torch.Tensor, labels:torch.Tensor,
            ) -> tuple[torch.Tensor, torch.Tensor]:
        outputs: torch.Tensor = self.model(inputs)
        loss: torch.Tensor = self.criterion(outputs, labels)
        return (outputs, loss)

    def computeMoeMetrics(
            self, currentInsigths:MoeExpertsInsigths|None,
            outputs:torch.Tensor, labels:torch.Tensor) -> MoeExpertsInsigths|None:
        return None

    def _getPBar(self, nb:int, name:str)->ProgressBar:
        return ProgressBar.simpleConfig(
            nb, name, newLineWhenFinished=False, updateEvery=self.updateEvery)



class TrainerClassif_MoE1(TrainerClassif):
    model:MoE_models.paper1.MOE_Model
    
    def __init__(
            self, model:MoE_models.paper1.MOE_Model, optimizer:torch.optim.Optimizer,
            criterion:torch.nn.Module, device:torch.device) -> None:
        super().__init__(model=model, optimizer=optimizer, criterion=criterion, device=device)
        self.__current_expertsOuts: numpy.ndarray|None = None
        self.__current_gating: numpy.ndarray|None = None
        
    def forwardAndLoss(
            self, inputs:torch.Tensor, labels:torch.Tensor
            )->tuple[torch.Tensor, torch.Tensor]:
        outputs, expertsOuts, gating = self.model(inputs)
        self.__current_expertsOuts = expertsOuts.detach().cpu().numpy()
        self.__current_gating = gating.detach().cpu().numpy()
        loss: torch.Tensor = self.model.applyLossCE(
            expertsOuts, gating, labels)
        return (outputs, loss)
    
    def computeMoeMetrics(
            self, currentInsigths:MoeExpertsInsigths|None,
            outputs:torch.Tensor, labels:torch.Tensor) -> MoeExpertsInsigths|None:
        assert self.__current_expertsOuts is not None
        assert self.__current_gating is not None
        batchSize, nbClasses, nbExperts  = self.__current_expertsOuts.shape
        if currentInsigths is None:
            currentInsigths = Moe1ExpertsInsigths.empty(
                nbExperts=nbExperts, nbClasses=nbClasses)
        assert isinstance(currentInsigths, Moe1ExpertsInsigths)
        npLabels = labels.cpu().numpy()
        npPredLabels = torch.argmax(outputs.detach(), dim=-1).cpu().numpy()
        predClassCount = numpy.bincount(npPredLabels, minlength=nbClasses)
        truthClassCount = numpy.bincount(npLabels, minlength=nbClasses)
        currentInsigths.sumExpertsGate += self.__current_gating.sum(axis=0)
        currentInsigths.nbPred += predClassCount
        currentInsigths.nbTruth += truthClassCount
        for iBatch in range(batchSize):
            gates = self.__current_gating[iBatch, :]
            currentInsigths.sumPredClassesExpertsGate[npPredLabels[iBatch], :] += gates
            currentInsigths.sumTrueClassesExpertsGate[npLabels[iBatch], :] += gates
        return currentInsigths