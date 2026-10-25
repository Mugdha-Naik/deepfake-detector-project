"""dfx_data.py: data loading for Phase 2. Lives at the REPO ROOT next to utils.py.

Reads faces through the frozen manifest: `face_rel` is relative to a faces root
(step5_faces for clean crops, compressed_faces for conditions B and D), so the same code
serves every condition.
"""
import os

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T


def resolve_dir(name, root="/kaggle/input"):
    """Find exactly one folder called `name` under `root` (does not descend into it)."""
    found = []
    for d, dirnames, _ in os.walk(root):
        if name in dirnames:
            found.append(os.path.join(d, name))
            dirnames.remove(name)
    if len(found) != 1:
        raise RuntimeError(f"expected exactly one '{name}' folder under {root}, found: {found}")
    return found[0]


def make_transform(mean, std, size, train):
    """Resize -> (train only: horizontal flip) -> tensor -> normalise.

    Deliberately NO compression or blur augmentation: that would train on the very shift
    the study measures.
    """
    ops = [T.Resize((size, size))]
    if train:
        ops.append(T.RandomHorizontalFlip(p=0.5))
    ops += [T.ToTensor(), T.Normalize(mean, std)]
    return T.Compose(ops)


class FaceDataset(Dataset):
    """Returns (image tensor, label float, face_id) so every prediction traces back to the manifest."""

    def __init__(self, df, faces_root, transform):
        self.paths = [os.path.join(faces_root, r) for r in df["face_rel"]]
        self.labels = df["label"].astype("float32").to_numpy()
        self.face_ids = df["face_id"].astype(str).tolist()
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        with Image.open(self.paths[i]) as im:
            x = self.transform(im.convert("RGB"))
        return x, torch.tensor(self.labels[i]), self.face_ids[i]


def make_loader(df, faces_root, mean, std, size, train, batch_size=32, num_workers=2, seed=0):
    ds = FaceDataset(df, faces_root, make_transform(mean, std, size, train))
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(ds, batch_size=batch_size, shuffle=train, num_workers=num_workers,
                      pin_memory=torch.cuda.is_available(), generator=g if train else None,
                      persistent_workers=num_workers > 0)


def denormalize(x, mean, std):
    """Undo Normalize for display. x: (3,H,W) tensor -> (H,W,3) numpy in [0,1]."""
    m = torch.tensor(mean).view(3, 1, 1)
    s = torch.tensor(std).view(3, 1, 1)
    return (x.cpu() * s + m).clamp(0, 1).permute(1, 2, 0).numpy()
