import pathlib
import PIL.Image as Image
from typing import (
    Callable, Protocol, Iterator, TypedDict, Literal, )

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import random_split

import torchvision
import torchvision.models.resnet as RESNET
from torchvision.transforms._presets import ImageClassification


DATASETS_ROOT = pathlib.Path("D:/AI_datas")

ImageTransform = Callable[[Image.Image], torch.Tensor]
# ImageClassification(**RESNET.ResNet50_Weights.IMAGENET1K_V2.transforms.keywords)


_DatasIterator = Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
"""yield (x, y, idx) tensors that are alredy on the device"""



class DatasIterFuncs(Protocol):
    def __call__(self, loader: DataLoader, device:torch.device) -> _DatasIterator:
        raise NotImplementedError

class CustomLoader():
    __slots__ = ("dataloader", "iterFunc", )
    def __init__(self, dataloader:DataLoader, iterFunc:DatasIterFuncs) -> None:
        self.dataloader = dataloader
        self.iterFunc = iterFunc
    
    def __len__(self)->int:
        return len(self.dataloader)
    
    def __call__(self, device:torch.device)->_DatasIterator:
        return self.iterFunc(self.dataloader, device=device)




class SizedDataset(Dataset):

    def __len__(self) -> int:
        raise NotImplementedError

    @staticmethod
    def iterDataloader(loader: DataLoader, device: torch.device)->_DatasIterator:
        """simplify the way to get the batched datas from dataloader\n
        yield (x, y, idx) tensors that are alredy on the device"""
        raise NotImplementedError


class ImageClass(int):
    ...


class ClassifDatasetOutput(TypedDict):
    image: torch.Tensor
    cls: ImageClass
    index: int

class ImagesClassifDataset(SizedDataset):

    def __init__(self, images: list[tuple[Image.Image, ImageClass]],
                 transform: ImageTransform | None = None):
        self.images: list[tuple[Image.Image, ImageClass]] = []
        for img, cls in images:
            self.images.append((img, cls))
        self.transformed: dict[int, tuple[torch.Tensor, ImageClass]] = {}
        self.transform: ImageTransform
        if transform is None:
            self.transform = torchvision.transforms.ToTensor()
        else:
            self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx)->ClassifDatasetOutput:
        assert isinstance(idx, int)
        idx = int(idx)
        transformed = self.transformed.get(idx, None)
        if transformed is not None:
            image, cls = transformed
        else:
            image, cls = self.images[idx]
            image = self.transform(image)
            self.transformed[idx] = (image, cls)
        return {'image': image, 'cls': cls, "index": idx}

    @staticmethod
    def iterDataloader(loader: DataLoader, device: torch.device)-> _DatasIterator:
        for sample in loader:
            yield (sample["image"].to(device), sample["cls"].to(device), sample["index"])



class HandleClassifDatas():
    def __init__(self, fullDataset: SizedDataset,
                 name: str, trainProp: float, nbClasses:int,
                 batchSizeTrain: int, batchSizeTest: int) -> None:
        """setup the test/train split and their dataloader based on 
            a given set of images to use (consider that the index are alredy offsetted)"""
        self.name: str = name
        self.nbClasses: int = nbClasses
        self.full_dataset = fullDataset
        nbSamplesTrain = int(len(self.full_dataset) * trainProp)
        self.dataset_train, self.dataset_test = random_split(
            self.full_dataset, lengths=[nbSamplesTrain, (len(self.full_dataset) - nbSamplesTrain)])
        self.datasLoader_train = DataLoader(self.dataset_train, batch_size=batchSizeTrain, shuffle=True, num_workers=0)
        self.datasLoader_test = DataLoader(self.dataset_test, batch_size=batchSizeTest, shuffle=True, num_workers=0)
        print(f"loaded {self.name}(total: {len(self.full_dataset)}), "
              f"train: {len(self.dataset_train)} [{len(self.datasLoader_train)} batches] | "
              f"test: {len(self.dataset_test)} [{len(self.datasLoader_test)} batches]")

    def iterDataloader(self, kind: Literal["train", "test"], device: torch.device):
        loader = (self.datasLoader_train if kind == "train" else self.datasLoader_test)
        return enumerate(self.full_dataset.iterDataloader(loader=loader, device=device))

    def train_cLoader(self)->CustomLoader:
        return CustomLoader(self.datasLoader_train, self.full_dataset.iterDataloader)
    def test_cLoader(self)->CustomLoader:
        return CustomLoader(self.datasLoader_test, self.full_dataset.iterDataloader)
    

class HandleImagesClassifDatas(HandleClassifDatas):
    full_dataset: ImagesClassifDataset
    
    def __init__(self, images: list[tuple[Image.Image, ImageClass]], name: str,
                 nbClasses: int, trainProp: float, batchSizeTrain: int, batchSizeTest: int) -> None:
        """setup the test/train split and their dataloader based on 
            a given set of images to use (consider that the index are alredy offsetted)"""
        super().__init__(
            fullDataset=ImagesClassifDataset(images=images, transform=None),
            name=name, trainProp=trainProp, nbClasses=nbClasses,
            batchSizeTrain=batchSizeTrain, batchSizeTest=batchSizeTest)

    @staticmethod
    def merge(*handlers:"HandleImagesClassifDatas", trainProp:float, 
              batchSizeTrain:int, batchSizeTest:int)->"HandleImagesClassifDatas":
        allImages: list[tuple[Image.Image, ImageClass]] = []
        currentNbClasses: int = 0
        for dataset in handlers:
            allImages.extend([
                (img, ImageClass(clsIndex + currentNbClasses))
                for (img, clsIndex) in dataset.full_dataset.images])
            currentNbClasses += dataset.nbClasses
        return HandleImagesClassifDatas(
            images=allImages, nbClasses=currentNbClasses,
            name=f"Merged[{', '.join([dts.name for dts in handlers])}]",
            trainProp=trainProp, batchSizeTrain=batchSizeTrain,
            batchSizeTest=batchSizeTest)


    

class MNIST_Datas(HandleImagesClassifDatas):
    def __init__(self, fromTrainSource: bool|None, maxSamples: int | None,
                 trainProp: float,
                 batchSizeTrain: int, batchSizeTest: int) -> None:
        self.fromTrainSource: bool|None = fromTrainSource
        images: list[tuple[Image.Image, ImageClass]] = []
        for src in ([True, False] if fromTrainSource is None else [fromTrainSource]):
            datas = torchvision.datasets.MNIST(DATASETS_ROOT, train=src, download=True)
            for (img, clsIndex) in datas:
                if (maxSamples is not None) and (len(images) >= maxSamples):
                    break
                images.append((img, ImageClass(clsIndex)))
        super().__init__(
            images=images, name="MNIST", trainProp=trainProp, nbClasses=10,
            batchSizeTrain=batchSizeTrain, batchSizeTest=batchSizeTest)


class FashionMNIST_Datas(HandleImagesClassifDatas):
    def __init__(self, fromTrainSource: bool|None, maxSamples: int | None,
                 trainProp: float, batchSizeTrain: int, batchSizeTest: int) -> None:
        self.fromTrainSource: bool|None = fromTrainSource
        images: list[tuple[Image.Image, ImageClass]] = []
        for src in ([True, False] if fromTrainSource is None else [fromTrainSource]):
            datas = torchvision.datasets.FashionMNIST(DATASETS_ROOT, train=src, download=True)
            for (img, clsIndex) in datas:
                if (maxSamples is not None) and (len(images) >= maxSamples):
                    break
                images.append((img, ImageClass(clsIndex)))
        super().__init__(
            images=images, name="FashionMNIST", trainProp=trainProp, nbClasses=10,
            batchSizeTrain=batchSizeTrain, batchSizeTest=batchSizeTest)

class Cifar10_Datas(HandleImagesClassifDatas):
    def __init__(self, fromTrainSource: bool|None, maxSamples: int | None,
                 trainProp: float, batchSizeTrain: int, batchSizeTest: int) -> None:
        self.fromTrainSource: bool|None = fromTrainSource
        images: list[tuple[Image.Image, ImageClass]] = []
        for src in ([True, False] if fromTrainSource is None else [fromTrainSource]):
            datas = torchvision.datasets.CIFAR10(DATASETS_ROOT, train=src, download=True)
            for (img, clsIndex) in datas:
                if (maxSamples is not None) and (len(images) >= maxSamples):
                    break
                images.append((img, ImageClass(clsIndex)))
        super().__init__(
            images=images, name="Cifar10", trainProp=trainProp, nbClasses=10,
            batchSizeTrain=batchSizeTrain, batchSizeTest=batchSizeTest)

class Cifar100_Datas(HandleImagesClassifDatas):
    def __init__(self, fromTrainSource: bool|None, maxSamples: int | None,
                 trainProp: float, batchSizeTrain: int, batchSizeTest: int) -> None:
        self.fromTrainSource: bool|None = fromTrainSource
        images: list[tuple[Image.Image, ImageClass]] = []
        for src in ([True, False] if fromTrainSource is None else [fromTrainSource]):
            datas = torchvision.datasets.CIFAR100(DATASETS_ROOT, train=src, download=True)
            for (img, clsIndex) in datas:
                if (maxSamples is not None) and (len(images) >= maxSamples):
                    break
                images.append((img, ImageClass(clsIndex)))
        super().__init__(
            images=images, name="Cifar100", trainProp=trainProp, nbClasses=100,
            batchSizeTrain=batchSizeTrain, batchSizeTest=batchSizeTest)

