from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset

CLASS_NAMES = ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASS_NAMES)}

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_DATA_ROOT = _REPO_ROOT / "skin_cancer_data"


class SkinDataset(Dataset):
    """ISIC/HAM10000 image dataset backed by a metadata dataframe."""

    def __init__(self, dataframe, root_dir=None, transform=None):
        self.df = dataframe.reset_index(drop=True).copy()
        self.root_dir = Path(root_dir) if root_dir is not None else _DEFAULT_DATA_ROOT
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_id = str(row["image_id"])
        label_name = str(row["dx"]).upper()
        if label_name not in CLASS_TO_IDX:
            raise ValueError(f"Unknown class label: {label_name}")

        label = CLASS_TO_IDX[label_name]
        path = self.root_dir / label_name / f"{image_id}.jpg"
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")

        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label, image_id
