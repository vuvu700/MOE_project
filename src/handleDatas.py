import pathlib
import PIL.Image as Image
from typing import (
    Callable, Protocol, Iterator, TypedDict, Literal, )

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import random_split

import torchvision
from torchvision.transforms import InterpolationMode
import torchvision.transforms.v2 as Transforms


DATASETS_ROOT = pathlib.Path("D:/AI_datas")


_DataAugmentation = Literal[None, "basic", "basic+degrade", ]
ImageTransform = Callable[[Image.Image|torch.Tensor], torch.Tensor]
_ImageNette_split = Literal["train", "val"]
_ImageNette_size = Literal["full", "320px", "160px"]


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

    def iterDataloader(self, loader: DataLoader, device: torch.device)->_DatasIterator:
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

    def __init__(self, images:list[tuple[Image.Image, ImageClass]],
                 cachableTransform:"ImageTransform", 
                 firstTransform:"ImageTransform", 
                 finalTransform:"ImageTransform", ):
        self.images: list[tuple[Image.Image, ImageClass]] = []
        for img, cls in images:
            self.images.append((img, cls))
        self.cachedTransformed: dict[int, tuple[torch.Tensor, ImageClass]] = {}
        self.cachableTransform: "ImageTransform" = cachableTransform
        self.firstTransform: "ImageTransform" = firstTransform
        self.finalTransform: "ImageTransform" = finalTransform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx)->ClassifDatasetOutput:
        assert isinstance(idx, int)
        idx = int(idx)
        cached = self.cachedTransformed.get(idx, None)
        if cached is not None:
            image, cls = cached
        else:
            image, cls = self.images[idx]
            image = self.cachableTransform(image)
            self.cachedTransformed[idx] = (image, cls)
        image = self.firstTransform(image)
        return {'image': image, 'cls': cls, "index": idx}

    def iterDataloader(self, loader: DataLoader, device: torch.device)-> _DatasIterator:
        for sample in loader:
            images = sample["image"].to(device)
            images = self.finalTransform(images)
            yield (images, sample["cls"].to(device), sample["index"])



class HandleClassifDatas():
    def __init__(self, fullDataset: SizedDataset, name: str, 
                 trainProp: float, nbClasses:int,
                 batchSizeTrain: int, batchSizeTest: int) -> None:
        """setup the test/train split and their dataloader based on 
            a given set of images to use (consider that the index are alredy offsetted)"""
        self.name: str = name
        self.nbClasses: int = nbClasses
        self.full_dataset = fullDataset
        nbSamplesTrain = int(len(self.full_dataset) * trainProp)
        self.dataset_train, self.dataset_test = random_split(
            self.full_dataset, lengths=[nbSamplesTrain, (len(self.full_dataset) - nbSamplesTrain)])
        self.datasLoader_train = DataLoader(self.dataset_train, batch_size=batchSizeTrain, shuffle=True)
        self.datasLoader_test = DataLoader(self.dataset_test, batch_size=batchSizeTest, shuffle=True)
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
                 nbClasses: int, imagesCropSize:tuple[int, int], trainProp: float, 
                 batchSizeTrain: int, batchSizeTest: int, dataAugemnt:"_DataAugmentation") -> None:
        """setup the test/train split and their dataloader based on 
            a given set of images to use (consider that the index are alredy offsetted)"""
        self.imagesCropSize: tuple[int, int] = imagesCropSize
        toTensor = [Transforms.ToImage(), Transforms.ToDtype(torch.float32, scale=True), ]
        norm = Transforms.Normalize(mean=[0.5]*3, std=[0.22]*3)
        crop = Transforms.CenterCrop(self.imagesCropSize)
        resizeCrop = Transforms.RandomResizedCrop(self.imagesCropSize, scale=(0.65, 1.5),  ratio=(1, 1))
        basicTransforms = [
            Transforms.RandomHorizontalFlip(),
            Transforms.RandomRotation((-15, +15), interpolation=InterpolationMode.BILINEAR, fill=0),]
        if dataAugemnt is None:
            cachableTransform = Transforms.Compose([*toTensor, norm, crop])
            firstTransform = (lambda x:x)
            finalTransform = (lambda x:x)
        elif dataAugemnt == "basic":
            cachableTransform = Transforms.Compose(toTensor)
            firstTransform = resizeCrop
            finalTransform = Transforms.Compose([norm, *basicTransforms])
        elif dataAugemnt == "basic+degrade":
            cachableTransform = Transforms.Compose(toTensor)
            firstTransform = resizeCrop
            finalTransform = Transforms.Compose([
                Transforms.ColorJitter(
                    brightness=0.35, contrast=0.25, saturation=0.25, hue=0.03),
                norm, *basicTransforms])
        else: raise ValueError(f"unknown data augemntation: {dataAugemnt!r}")
        super().__init__(
            fullDataset=ImagesClassifDataset(
                images=images, cachableTransform=cachableTransform, 
                firstTransform=firstTransform, finalTransform=finalTransform),
            name=name, trainProp=trainProp, nbClasses=nbClasses, 
            batchSizeTrain=batchSizeTrain, batchSizeTest=batchSizeTest)

    @staticmethod
    def merge(*handlers:"HandleImagesClassifDatas", trainProp:float, 
              batchSizeTrain:int, batchSizeTest:int, 
              dataAugemnt:"_DataAugmentation")->"HandleImagesClassifDatas":
        lstImgCropSizes = [h.imagesCropSize for h in handlers]
        assert set(lstImgCropSizes) != 1, f"got different imagesCropSize: {lstImgCropSizes}"
        imagesCropSize = lstImgCropSizes[0]
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
            trainProp=trainProp, imagesCropSize=imagesCropSize,
            batchSizeTrain=batchSizeTrain, 
            batchSizeTest=batchSizeTest, dataAugemnt=dataAugemnt)


    

class MNIST_Datas(HandleImagesClassifDatas):
    def __init__(self, fromTrainSource: bool|None, maxSamples: int | None,
                 trainProp: float, batchSizeTrain: int, batchSizeTest: int, 
                 dataAugemnt:"_DataAugmentation"=None) -> None:
        self.fromTrainSource: bool|None = fromTrainSource
        images: list[tuple[Image.Image, ImageClass]] = []
        for src in ([True, False] if fromTrainSource is None else [fromTrainSource]):
            datas = torchvision.datasets.MNIST(DATASETS_ROOT, train=src, download=True)
            for (img, clsIndex) in datas:
                if (maxSamples is not None) and (len(images) >= maxSamples):
                    break
                images.append((img, ImageClass(clsIndex)))
        super().__init__(
            images=images, name="MNIST", trainProp=trainProp, 
            nbClasses=10, imagesCropSize=(28, 28), batchSizeTrain=batchSizeTrain,
            batchSizeTest=batchSizeTest, dataAugemnt=dataAugemnt)


class FashionMNIST_Datas(HandleImagesClassifDatas):
    def __init__(self, fromTrainSource: bool|None, maxSamples: int | None,
                 trainProp: float, batchSizeTrain: int, batchSizeTest: int, 
                 dataAugemnt:"_DataAugmentation"=None) -> None:
        self.fromTrainSource: bool|None = fromTrainSource
        images: list[tuple[Image.Image, ImageClass]] = []
        for src in ([True, False] if fromTrainSource is None else [fromTrainSource]):
            datas = torchvision.datasets.FashionMNIST(DATASETS_ROOT, train=src, download=True)
            for (img, clsIndex) in datas:
                if (maxSamples is not None) and (len(images) >= maxSamples):
                    break
                images.append((img, ImageClass(clsIndex)))
        super().__init__(
            images=images, name="FashionMNIST", trainProp=trainProp, 
            nbClasses=10, imagesCropSize=(32, 32), batchSizeTrain=batchSizeTrain, 
            batchSizeTest=batchSizeTest, dataAugemnt=dataAugemnt)

class Cifar10_Datas(HandleImagesClassifDatas):
    def __init__(self, fromTrainSource: bool|None, maxSamples: int | None,
                 trainProp: float, batchSizeTrain: int, batchSizeTest: int, 
                 dataAugemnt:"_DataAugmentation"=None) -> None:
        self.fromTrainSource: bool|None = fromTrainSource
        images: list[tuple[Image.Image, ImageClass]] = []
        for src in ([True, False] if fromTrainSource is None else [fromTrainSource]):
            datas = torchvision.datasets.CIFAR10(DATASETS_ROOT, train=src, download=True)
            for (img, clsIndex) in datas:
                if (maxSamples is not None) and (len(images) >= maxSamples):
                    break
                images.append((img, ImageClass(clsIndex)))
        super().__init__(
            images=images, name="Cifar10", trainProp=trainProp, 
            nbClasses=10, imagesCropSize=(32, 32), batchSizeTrain=batchSizeTrain,
            batchSizeTest=batchSizeTest, dataAugemnt=dataAugemnt)

class Cifar100_Datas(HandleImagesClassifDatas):
    def __init__(self, fromTrainSource: bool|None, maxSamples: int | None,
                 trainProp: float, batchSizeTrain: int, batchSizeTest: int, 
                 dataAugemnt:"_DataAugmentation"=None) -> None:
        self.fromTrainSource: bool|None = fromTrainSource
        images: list[tuple[Image.Image, ImageClass]] = []
        for src in ([True, False] if fromTrainSource is None else [fromTrainSource]):
            datas = torchvision.datasets.CIFAR100(DATASETS_ROOT, train=src, download=True)
            for (img, clsIndex) in datas:
                if (maxSamples is not None) and (len(images) >= maxSamples):
                    break
                images.append((img, ImageClass(clsIndex)))
        super().__init__(
            images=images, name="Cifar100", trainProp=trainProp, 
            nbClasses=100, imagesCropSize=(32, 32), batchSizeTrain=batchSizeTrain,
            batchSizeTest=batchSizeTest, dataAugemnt=dataAugemnt)

class ImageNette_Datas(HandleImagesClassifDatas):
    def __init__(self, fromTrainSource: "_ImageNette_split|None", size:"_ImageNette_size",
                 maxSamples:int|None, trainProp: float, batchSizeTrain: int, 
                 batchSizeTest: int, dataAugemnt:"_DataAugmentation"=None) -> None:
        self.fromTrainSource: "_ImageNette_split|None" = fromTrainSource
        imagesCropSize = {"full": (500, 500), "320px": (320, 320), "160px": (160, 160)}[size]
        images: list[tuple[Image.Image, ImageClass]] = []
        for src in (("train", "val") if fromTrainSource is None else [fromTrainSource]):
            datas = torchvision.datasets.Imagenette(
                DATASETS_ROOT, split=src, size=size, download=True)
            for (img, clsIndex) in datas:
                if (maxSamples is not None) and (len(images) >= maxSamples):
                    break
                images.append((img, ImageClass(clsIndex)))
        super().__init__(
            images=images, name=f"imageNette({size})", trainProp=trainProp, 
            nbClasses=10, imagesCropSize=imagesCropSize, batchSizeTrain=batchSizeTrain,
            batchSizeTest=batchSizeTest, dataAugemnt=dataAugemnt)

class ImageNet_Datas(HandleImagesClassifDatas):
    def __init__(self, fromTrainSource: "_ImageNette_split|None", size:"_ImageNette_size",
                 maxSamples:int|None, trainProp: float, batchSizeTrain: int, 
                 batchSizeTest: int, dataAugemnt:"_DataAugmentation"=None) -> None:
        raise NotImplementedError
        self.fromTrainSource: "_ImageNette_split|None" = fromTrainSource
        imagesCropSize = {"full": (500, 500), "320px": (320, 320), "160px": (160, 160)}[size]
        images: list[tuple[Image.Image, ImageClass]] = []
        for src in (("train", "val") if fromTrainSource is None else [fromTrainSource]):
            datas = torchvision.datasets.ImageNet(
                DATASETS_ROOT, split=src, size=size, download=True)
            for (img, clsIndex) in datas:
                if (maxSamples is not None) and (len(images) >= maxSamples):
                    break
                images.append((img, ImageClass(clsIndex)))
        super().__init__(
            images=images, name=f"imageNette({size})", trainProp=trainProp, 
            nbClasses=10, imagesCropSize=imagesCropSize, batchSizeTrain=batchSizeTrain,
            batchSizeTest=batchSizeTest, dataAugemnt=dataAugemnt)

