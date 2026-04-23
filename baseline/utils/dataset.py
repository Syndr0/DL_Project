from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image
from torch.utils.data import Dataset


def build_dataset_split(data_dir, train_ratio: float = 0.7):
    """Split an ImageFolder-style directory into gallery and query sets.

    Per class: first `train_ratio` images → gallery/train,
               remaining images        → query.
    Images within each class are sorted by filename for reproducibility.

    Args:
        data_dir:    Root directory whose subdirectories are class names.
        train_ratio: Fraction of each class to put in the gallery (default 0.7).

    Returns:
        train_paths  : list[Path]   gallery image paths
        train_labels : np.ndarray   integer class labels for gallery
        query_paths  : list[Path]   query image paths
        query_labels : np.ndarray   integer class labels for query
        classes      : list[str]    sorted class names (index == label)
    """
    data_dir = Path(data_dir)
    classes  = sorted(d.name for d in data_dir.iterdir() if d.is_dir())
    cls2idx  = {c: i for i, c in enumerate(classes)}

    train_paths, train_labels = [], []
    query_paths, query_labels = [], []

    for cls in classes:
        imgs  = sorted((data_dir / cls).glob('*.jpg'))
        split = int(len(imgs) * train_ratio)
        train_paths  += imgs[:split]
        train_labels += [cls2idx[cls]] * split
        query_paths  += imgs[split:]
        query_labels += [cls2idx[cls]] * (len(imgs) - split)

    return (train_paths, np.array(train_labels),
            query_paths, np.array(query_labels),
            classes)


def build_stanford_train(data_dir):
    """Load SOP train split (Ebay_train.txt) for contrastive fine-tuning.

    Classes are completely disjoint from the test split, making this a proper
    metric learning setup: fine-tune on train classes, evaluate on unseen test classes.

    Args:
        data_dir: Root containing Ebay_train.txt and image subdirectories.

    Returns:
        train_paths  : list[Path]
        train_labels : np.ndarray  (0-indexed integer class labels)
        classes      : list[str]   (index == label)
    """
    data_dir = Path(data_dir)
    paths, raw_labels = [], []

    with open(data_dir / 'Ebay_train.txt') as f:
        next(f)
        for line in f:
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            img_path = data_dir / parts[3]
            if img_path.exists():
                paths.append(img_path)
                raw_labels.append(int(parts[1]))

    all_cls = sorted(set(raw_labels))
    cls2idx = {c: i for i, c in enumerate(all_cls)}
    labels  = np.array([cls2idx[l] for l in raw_labels])
    classes = [str(c) for c in all_cls]

    return paths, labels, classes


def build_stanford_split(data_dir, train_ratio: float = 0.7):
    """Load Stanford Online Products dataset for retrieval evaluation.

    Train and test sets have completely disjoint class IDs, so this function
    uses only the test split and applies the same per-class gallery/query
    split as build_dataset_split: first train_ratio images → gallery,
    remaining → query.  Images within each class are sorted by filename.

    Args:
        data_dir:    Root containing Ebay_test.txt and image subdirectories.
        train_ratio: Fraction of each class to put in the gallery (default 0.7).

    Returns:
        gallery_paths  : list[Path]
        gallery_labels : np.ndarray  (0-indexed integer class labels)
        query_paths    : list[Path]
        query_labels   : np.ndarray
        classes        : list[str]   (index == label)
    """
    from collections import defaultdict

    data_dir = Path(data_dir)

    cls_images = defaultdict(list)
    with open(data_dir / 'Ebay_test.txt') as f:
        next(f)
        for line in f:
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            img_path = data_dir / parts[3]
            if img_path.exists():
                cls_images[int(parts[1])].append(img_path)

    all_cls = sorted(cls_images.keys())
    cls2idx = {c: i for i, c in enumerate(all_cls)}

    gallery_paths, gallery_labels = [], []
    query_paths,   query_labels   = [], []

    for cls in all_cls:
        imgs  = sorted(cls_images[cls])
        idx   = cls2idx[cls]
        split = max(1, int(len(imgs) * train_ratio))
        for img in imgs[:split]:
            gallery_paths.append(img)
            gallery_labels.append(idx)
        for img in imgs[split:]:
            query_paths.append(img)
            query_labels.append(idx)

    return (gallery_paths, np.array(gallery_labels),
            query_paths,   np.array(query_labels),
            [str(c) for c in all_cls])


class _ImgDataset(Dataset):
    """Minimal image dataset for fine-tuning DataLoader."""

    def __init__(self, paths: Sequence, labels: Sequence, transform):
        self.paths   = list(paths)
        self.labels  = list(labels)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert('RGB')
        return self.transform(img), int(self.labels[i])
