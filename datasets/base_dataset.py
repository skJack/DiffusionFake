from __future__ import annotations

import copy
from collections import OrderedDict
from pathlib import Path
from typing import List, Set, Tuple

import cv2
from torch.utils.data import Dataset, ConcatDataset


class BaseDataset(Dataset):
    def __init__(self,
                 path: str | Path,
                 label: int,
                 index: Set[int] | str | Path = None,
                 suffixes: Tuple[str] = ("jpg", "png", "jpeg"),
                 transform=None,
                 target_transform=None):
        """
        :param path: The path to the directory containing the dataset images
        :param label: The label or class index associated with the dataset.
        :param index: An optional parameter that specifies a subset of indices to load from the dataset. It can be
            a set of integers, a path to a file containing the indices, or None to load all indices.
        :param suffixes: A tuple of file extensions to consider when loading images from the dataset directory. Only
            files with these extensions will be loaded.
        :param transform: An optional transformation function to apply to the loaded images.
        :param target_transform: An optional transformation function to apply to the labels.
        """
        self.path = Path(path)
        self.label = label
        self.index = index
        if isinstance(index, str) or isinstance(index, Path):
            self.index = self._load_index(index)
        self.suffixes = suffixes
        self.transform = transform
        self.target_transform = target_transform
        # Load items
        self.ids, self.data = self._load_items()
        self.control = True

    def _load_index(self, index_path: str | Path) -> Set[int]:
        setattr(self, "index_path", index_path)
        index_path = Path(index_path)
        assert index_path.exists(), f"Path {index_path} does not exist."
        if index_path.suffix == ".pickle":
            import pickle
            index_list = pickle.load(open(index_path, "rb"))
            index_set = set([int(x) for x in index_list])  # Turn list into set for O(1) search.
            return index_set
        else:
            raise NotImplementedError(f"Index file {index_path} is not supported.")

    def _load_items(self) -> Tuple[List[int], OrderedDict[int, Path]]:
        files = []
        for suffix in self.suffixes:
            files.extend(list(self.path.glob(f"*.{suffix}")))

        ids = sorted([int(f.stem) for f in files])
        files = sorted(files, key=lambda x: int(x.stem))

        data = OrderedDict()
        if self.index is not None:
            try:
                data = OrderedDict([(ids[i], files[i]) for i in range(len(ids)) if ids[i] in self.index])
            except IndexError as e:
                raise IndexError(f"The index {set(self.index) - set(ids)} is out of range in {self.path}.") from e
        else:
            data = OrderedDict(zip(ids, files))

        return list(data.keys()), data

    def __getitem__(self, index: int) -> Tuple:
        image_id = self.ids[index]
        image_path = self.get_image_by_id(image_id)

        # Load image
        image = cv2.imread(image_path)
        assert image is not None, f"Image {image_path} is not valid."
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = self.transform(image=image)["image"] if self.transform is not None else image
        target = self.target_transform(self.label) if self.target_transform is not None else self.label
        if self.control == True:
            return{
                'txt': '',  # 如果没有对应的文本提示,可以留空或设置为空字符串
                # 'hint': image,
                'hint': image,
                'label': target,
                'path' : image_path
            }

        return image, target, str(self.path), image_path

    def get_image_by_id(self, _id: int) -> str:
        return str(self.data[_id])

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        repr = f"{self.__class__.__name__}("
        repr += f"\n    path={self.path},"
        repr += f"\n    label={self.label},"
        repr += f"\n    length={len(self)},"
        repr += f"\n    index_path={getattr(self, 'index_path', 'None')},"
        repr += f"\n    suffixes={self.suffixes} \n"
        return repr


class MixDataset(Dataset):
    def __init__(self,
                 real_datasets: List[BaseDataset] | List[str] | List[Path],
                 fake_datasets: List[BaseDataset] | List[str] | List[Path],
                 index: Set[int] | str | Path = None,
                 suffixes: Tuple[str] = ("jpg", "png", "jpeg"),
                 transform=None,
                 target_transform=None):

        """
        :param real_datasets: A list of real datasets, each of which can be a path to the dataset directory, or a
            BaseDataset object.
        :param fake_datasets: A list of fake datasets, each of which can be a path to the dataset directory, or a
            BaseDataset object.
        :param index: When specified, only the indices in this set will be loaded from the dataset.
        :param suffixes: A tuple of file extensions to consider when loading images from the dataset directory.
        :param transform: If specified, the transformation will be overridden for all datasets.
        :param target_transform: If specified, the transformation will be overridden for all datasets.
        """

        self.real_datasets = []
        for i in range(len(real_datasets)):
            real_dataset: BaseDataset | str | Path = real_datasets[i]
            if isinstance(real_dataset, str) or isinstance(real_dataset, Path):
                self.real_datasets.append(BaseDataset(real_dataset, label=0, index=index, suffixes=suffixes,
                                                      transform=transform, target_transform=target_transform))
            elif isinstance(real_dataset, BaseDataset):
                assert real_dataset.label == 0, f"Label of real dataset: \n{real_dataset} is not 0."
                self.real_datasets.append(
                    self._reload_items(real_dataset, index, suffixes, transform, target_transform))
            else:
                raise NotImplementedError(f"Dataset {real_dataset} is not supported.")

        self.fake_datasets = []
        for i in range(len(fake_datasets)):
            fake_dataset: BaseDataset | str | Path = fake_datasets[i]
            if isinstance(fake_dataset, str) or isinstance(fake_dataset, Path):
                self.fake_datasets.append(BaseDataset(fake_dataset, label=1, index=index, suffixes=suffixes,
                                                      transform=transform, target_transform=target_transform))
            elif isinstance(fake_dataset, BaseDataset):
                assert fake_dataset.label != 0, f"Label of fake dataset: \n{fake_dataset} is 0."
                self.fake_datasets.append(
                    self._reload_items(fake_dataset, index, suffixes, transform, target_transform))
            else:
                raise NotImplementedError(f"Dataset {fake_dataset} is not supported.")

        self.concat_datasets = ConcatDataset(self.real_datasets + self.fake_datasets)

        self.index = index
        self.suffixes = suffixes
        self.transform = transform
        self.target_transform = target_transform

    def _reload_items(self, dataset: BaseDataset, index: Set[int] | str | Path, suffixes: Tuple[str],
                      transform, target_transform):
        dataset = copy.deepcopy(dataset)  # copy dataset to avoid changing the original dataset
        if isinstance(index, str) or isinstance(index, Path):
            dataset.index = dataset._load_index(index)
        dataset.suffixes = suffixes
        # Overwrite transform and target_transform if not None.
        dataset.transform = transform if dataset.transform is None else dataset.transform
        dataset.target_transform = target_transform if dataset.target_transform is None else dataset.target_transform
        # Reload items
        dataset.ids, dataset.data = dataset._load_items()
        return dataset

    def get_image_by_id(self, _id: int) -> Tuple[List[str], List[str]]:
        real_images = [d.get_image_by_id(_id) for d in self.real_datasets]
        fake_images = [d.get_image_by_id(_id) for d in self.fake_datasets]
        return real_images, fake_images

    def __getitem__(self, index: int) -> Tuple:
        return self.concat_datasets[index]

    def __len__(self):
        return len(self.concat_datasets)

    def __repr__(self):
        repr = f"{self.__class__.__name__}("
        repr += f"\n    real_datasets: "
        for dataset in self.real_datasets:
            repr += "\n        " + dataset.__repr__().replace("\n", "\n        ")
        repr += f"\n    fake_datasets: "
        for dataset in self.fake_datasets:
            repr += f"\n        " + dataset.__repr__().replace("\n", "\n        ")
        return repr


if __name__ == "__main__":
    train_dataset = BaseDataset(
        path="/media/kesun/Dataset/Face/MM_CelebA/CelebAMask-HQ/CelebA-HQ-img",
        label=0,
        index="/media/kesun/Dataset/Face/MM_CelebA/train/filenames.pickle",
    )

    print(train_dataset)

    total_dataset = BaseDataset(
        path="/media/kesun/Dataset/Face/MM_CelebA/CelebAMask-HQ/CelebA-HQ-img",
        label=1
    )
    print(total_dataset)

    mix_dataset = MixDataset(
        real_datasets=[train_dataset],
        fake_datasets=[total_dataset, total_dataset],
        index="/media/kesun/Dataset/Face/MM_CelebA/test/filenames.pickle",
    )
    print(mix_dataset)