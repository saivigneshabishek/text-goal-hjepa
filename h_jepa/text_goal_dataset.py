import h5py
import hdf5plugin  # noqa: F401
import numpy as np
from pathlib import Path
import torch
from torch.utils.data import Dataset

class GoalEmbeddings:
    """
    Lookup for precomputed goal text embeddings.
        store = GoalEmbeddings.from_npz(Path)
        store["Go to maze cell (1, 1)."] # (D,)
        store[(1, 1)]                    # (M, D)
    """

    def __init__(self, embeddings, texts, cells):
        self.embeddings = embeddings.astype(np.float32, copy=False) #(N, D)
        self._by_text = {t: i for i, t in enumerate(texts)}
        self._by_cell = {}

        for i, (r, c) in enumerate(cells.tolist()):
            self._by_cell.setdefault((int(r), int(c)), []).append(i)
    
    @classmethod
    def from_npz(cls, path):
        with np.load(Path(path), allow_pickle=False) as records:
            return cls(records["embeddings"], records["texts"].tolist(), records["cells"])
    
    def __getitem__(self, key):
        if isinstance(key, str):
            return self.embeddings[self._by_text[key]] #(D,)
        if isinstance(key, tuple):
            return self.embeddings[self._by_cell[tuple(key)]] #(M, D)
        raise TypeError(f"Unsupported key: {type(key)} / Supported keys: str(text), (row, col)")
    
    def __len__(self):
        return len(self._by_text)

def get_cell_pos(xy):
    xy = np.asarray(xy, dtype=np.float64)
    if xy.ndim == 0 or xy.shape[-1] != 2:
        raise ValueError(f"Expected shape (..., 2), got {xy.shape}")
    if not np.isfinite(xy).all():
        raise ValueError("Positions must contain only finite values")

    cell_size = 4.0
    offset = 4.0
    x = xy[..., 0]
    y = xy[..., 1]

    row = (y + offset + (cell_size/2)) / cell_size
    col = (x + offset + (cell_size/2)) / cell_size

    return np.stack([row, col], axis=-1).astype(np.int32)

class GoalTextAlignmentDataset(Dataset):
    def __init__(self, data_dir="data", train=True):

        self.split = "train" if train else "eval"
        self.data_path = Path(data_dir)/f"visual_antmaze_medium_probing_{self.split}.h5"
        self.text_path = Path(data_dir)/"antmaze_text_embeddings"/f"{self.split}_text_embeddings.npz"
        self.text_embeddings = GoalEmbeddings.from_npz(self.text_path)

        with h5py.File(self.data_path, "r") as f:
            self.length = len(f["pixels"])

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        with h5py.File(self.data_path, "r") as f:
            img = f["pixels"][idx]
            xy = f["xy"][idx]
        cell = get_cell_pos(xy)
        text_embeddings = torch.from_numpy(self.text_embeddings[tuple(cell)])
        img = torch.from_numpy(img).permute(2, 0, 1)
        
        return img, text_embeddings