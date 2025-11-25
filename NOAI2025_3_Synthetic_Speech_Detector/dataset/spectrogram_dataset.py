import os
import torch
from torch.utils.data import Dataset


class SpectrogramDataset(Dataset):
    """
    Load spectrogram data from preprocessed .pt files.

    For training_set/, assumes:
        dataset/training_set/
            bonafide/
            spoof/

    For validation_set/ and testing_set/, assumes:
        dataset/validation_set/    (all .pt files in this folder, no subfolders)
        dataset/testing_set/       (all .pt files in this folder, no subfolders)

    No label will be provided for val/test sets to prevent label leakage.
    """

    def __init__(self, directory):
        self.samples = []

        if "training" in directory:
            label_map = {"bonafide": 0, "spoof": 1}
            for label_name, label in label_map.items():
                label_dir = os.path.join(directory, label_name)
                if not os.path.isdir(label_dir):
                    continue
                for fname in os.listdir(label_dir):
                    if fname.endswith(".pt"):
                        self.samples.append(
                            {"path": os.path.join(label_dir, fname), "label": label}
                        )
        else:
            for fname in sorted(os.listdir(directory)):
                if fname.endswith(".pt"):
                    self.samples.append({"path": os.path.join(directory, fname)})

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        spec = torch.load(item["path"])
        out = {"spectrogram": spec}
        if "label" in item:
            out["label"] = torch.tensor(item["label"], dtype=torch.long)
        return out
