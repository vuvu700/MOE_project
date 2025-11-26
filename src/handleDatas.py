import typing
import pathlib
import PIL.Image as Image

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import random_split

import torchvision
import torchvision.models.resnet as RESNET
from torchvision.transforms._presets import ImageClassification


DATASETS_ROOT = pathlib.Path("./datas/")

ImageTransform = typing.Callable[[Image.Image], torch.Tensor]
# ImageClassification(**RESNET.ResNet50_Weights.IMAGENET1K_V2.transforms.keywords)


class SizedDataset(Dataset):

    def __len__(self) -> int:
        raise NotImplementedError

    @staticmethod
    def iterDataloader(loader: DataLoader, device: torch.device) \
            -> typing.Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """simplify the way to get the batched datas from dataloader\n
        yield (x, y, idx) tensors that are alredy on the device"""
        raise NotImplementedError


class ImageClass(int):
    ...


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

    def __getitem__(self, idx):
        assert isinstance(idx, int)
        idx = int(idx)
        transformed = self.transformed.get(idx, None)
        if transformed is not None:
            image, cls = transformed
        else:
            image, cls = self.images[idx]
            image = self.transform(image)
            self.transformed[idx] = (image, cls)
        return {'image': image, 'class': cls, "index": idx}

    @staticmethod
    def iterDataloader(loader: DataLoader, device: torch.device) \
            -> typing.Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        for sample in loader:
            yield (sample["image"].to(device), sample["class"].to(device), sample["index"])


def iterImagesDataset(loader: DataLoader, device: torch.device,
                      ) -> typing.Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    for sample in loader:
        yield (sample["image"].to(device), sample["class"].to(device), sample["index"])


class HandleClassifDatas():
    def __init__(self, fullDataset: SizedDataset,
                 name: str, trainProp: float, classesOffset: int,
                 batchSizeTrain: int, batchSizeTest: int) -> None:
        # TODO
        """setup the test/train split and their dataloader based on 
            a given set of images to use (consider that the index are alredy offsetted)"""
        self.name: str = name
        self.classesOffset: int = classesOffset
        self.full_dataset = fullDataset
        nbSamplesTrain = int(len(self.full_dataset) * trainProp)
        self.dataset_train, self.dataset_test = random_split(
            self.full_dataset, lengths=[nbSamplesTrain, (len(self.full_dataset) - nbSamplesTrain)])
        self.datasLoader_train = DataLoader(self.dataset_train, batch_size=batchSizeTrain, shuffle=True, num_workers=0)
        self.datasLoader_test = DataLoader(self.dataset_test, batch_size=batchSizeTest, shuffle=True, num_workers=0)
        print(f"loaded {self.name}(total: {len(self.full_dataset)}), "
              f"train: {len(self.dataset_train)} [{len(self.datasLoader_train)} batches] | "
              f"test: {len(self.dataset_test)} [{len(self.datasLoader_test)} batches]")

    def iterDataloader(self, kind: typing.Literal["train", "test"], device: torch.device):
        loader = (self.datasLoader_train if kind == "train" else self.datasLoader_test)
        return enumerate(self.full_dataset.iterDataloader(loader=loader, device=device))


class HandleImagesClassifDatas(HandleClassifDatas):
    def __init__(self, images: list[tuple[Image.Image, ImageClass]],
                 name: str, trainProp: float, classesOffset: int,
                 batchSizeTrain: int, batchSizeTest: int) -> None:
        """setup the test/train split and their dataloader based on 
            a given set of images to use (consider that the index are alredy offsetted)"""
        self.full_dataset: ImagesClassifDataset
        super().__init__(
            fullDataset=ImagesClassifDataset(images=images, transform=None),
            name=name, trainProp=trainProp, classesOffset=classesOffset,
            batchSizeTrain=batchSizeTrain, batchSizeTest=batchSizeTest)


class MNIST_Datas(HandleImagesClassifDatas):

    def __init__(self, fromTrainSource: bool, maxSamples: int | None,
                 trainProp: float, classesOffset: int,
                 batchSizeTrain: int, batchSizeTest: int) -> None:
        self.fromTrainSource: bool = fromTrainSource
        mnist = torchvision.datasets.mnist.MNIST(
            root=DATASETS_ROOT, train=self.fromTrainSource, download=True)
        images: list[tuple[Image.Image, ImageClass]] = []
        for (img, cls) in mnist:
            if (maxSamples is not None) and (len(images) >= maxSamples):
                break
            images.append((img, ImageClass(cls + classesOffset)))
        super().__init__(
            images=images, name="MNIST", trainProp=trainProp, classesOffset=classesOffset,
            batchSizeTrain=batchSizeTrain, batchSizeTest=batchSizeTest)
