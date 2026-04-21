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


def build_stanford_split(data_dir):
    """Load Stanford Online Products dataset using official train/test split.

    Reads Ebay_train.txt (gallery) and Ebay_test.txt (query) from data_dir.
    Images live in category subdirectories referenced by the text files.

    Args:
        data_dir: Root containing Ebay_train.txt, Ebay_test.txt, and image dirs.

    Returns:
        gallery_paths  : list[Path]
        gallery_labels : np.ndarray  (0-indexed integer class labels)
        query_paths    : list[Path]
        query_labels   : np.ndarray
        classes        : list[str]   (index == label)
    """
    data_dir = Path(data_dir)

    def _load_txt(txt_path):
        paths, labels = [], []
        with open(txt_path) as f:
            next(f)  # skip header line
            for line in f:
                parts = line.strip().split()
                if len(parts) < 4:
                    continue
                img_path = data_dir / parts[3]
                if img_path.exists():
                    paths.append(img_path)
                    labels.append(int(parts[1]))
        return paths, labels

    gallery_paths, gallery_raw = _load_txt(data_dir / 'Ebay_train.txt')
    query_paths,   query_raw   = _load_txt(data_dir / 'Ebay_test.txt')

    all_cls = sorted(set(gallery_raw) | set(query_raw))
    cls2idx = {c: i for i, c in enumerate(all_cls)}

    gallery_labels = np.array([cls2idx[l] for l in gallery_raw])
    query_labels   = np.array([cls2idx[l] for l in query_raw])
    classes        = [str(c) for c in all_cls]

    return gallery_paths, gallery_labels, query_paths, query_labels, classes


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
