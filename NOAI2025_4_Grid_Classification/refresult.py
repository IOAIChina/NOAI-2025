import os
import zipfile
import numpy as np
import pandas as pd
from PIL import Image
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet34
from torchvision import transforms
import matplotlib.pyplot as plt
import random

from torchvision.models import (
    resnet18,
    resnet34,
    resnet50,
    ResNet18_Weights,
    ResNet34_Weights,
    ResNet50_Weights,
)

import warnings
from collections import defaultdict

# ignore only the torchvision.io.image UserWarning about loading the C extension
warnings.filterwarnings(
    "ignore",
    message="Failed to load image Python extension:.*",
    category=UserWarning,
    module=r"torchvision\.io\.image"
)

# ============= Provided two reference solutions ===========
# 1. DL based method: CNN
# 2. Feature Engineering: Canny extraction

# ===================== Utilities =====================
def save_csv(preds, fname):
    pd.DataFrame(preds).to_csv(fname, index=False, header=False)

def detect_collage_with_thresh(image, threshold,
                               canny1=100, canny2=200, line_size=5):
    npimg = np.array(image.convert("RGB"))
    gray  = cv2.cvtColor(npimg, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, canny1, canny2)

    kh = cv2.getStructuringElement(cv2.MORPH_RECT, (line_size, 1))
    kv = cv2.getStructuringElement(cv2.MORPH_RECT, (1, line_size))
    horiz = cv2.erode(edges, kh)
    vert  = cv2.erode(edges, kv)

    hprof = horiz.sum(axis=1)
    vprof = vert.sum(axis=0)
    peak  = max(hprof.max(), vprof.max())
    return int(peak > threshold), peak
def preprocess_edge_map(image):
    img_np = np.array(image.resize((224, 224)))
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 100, 200)

    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))
    kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))
    horiz = cv2.erode(edges, kernel_h)
    vert = cv2.erode(edges, kernel_v)

    # Construct RGB [H, V, E] and scale
    rgb = np.stack([horiz, vert, edges], axis=-1).astype(np.float32) / 255.0
    tensor = torch.tensor(rgb).permute(2, 0, 1)  # [3, 224, 224]

    # Normalize using ImageNet mean/std
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    normalized = (tensor - mean) / std

    return normalized
def visualize_edge_map(img_path):
    img = Image.open(img_path).convert("RGB")
    edge_tensor = preprocess_edge_map(img)  # shape: [1, H, W]
    edge_map = edge_tensor.squeeze(0).numpy()  # to 2D array

    plt.figure(figsize=(6, 6))
    plt.imshow(edge_map, cmap="gray")
    plt.title("Preprocessed Edge Map")
    plt.axis("off")
    plt.show()


    def __init__(self, data_dir, train_file, mode="train", train_manual=None):
        self.images = []
        self.labels = []
        self.mode = mode
        self.data_dir = data_dir

        # --- Load from train_manual.csv (base labels) ---
        if train_manual:
            with open(train_manual, "r", encoding="utf-8") as file:
                for line in file:
                    fields = [x.strip() for x in line.strip().split(",")]
                    if len(fields) != 2:
                        continue
                    img_name, label_str = fields
                    label = int(label_str)
                    path = os.path.join(data_dir, f"{img_name}.jpg")
                    if not os.path.exists(path):
                        path = os.path.join(data_dir, f"{img_name}.png")
                    self.images.append(path)
                    self.labels.append(label)

        # --- Load extra augmentation images from train_file.csv ---
        if train_file and self.mode == "train":
            augment_image_paths = []
            augment_labels = []
            with open(train_file, "r", encoding="gbk") as file:
                for line in file:
                    fields = [f.strip() for f in line.strip().split(",")]
                    if len(fields) != 4:
                        continue
                    img_name, _, _, _ = fields
                    path = os.path.join(data_dir, f"{img_name}.jpg")
                    if not os.path.exists(path):
                        path = os.path.join(data_dir, f"{img_name}.png")
                    augment_image_paths.append(path)
                    augment_labels.append(-1)  # label ignored during generation

            # Apply augmentation and save to disk
            aug_imgs, aug_labels = self.augmentation(augment_image_paths, augment_labels)
            aug_dir = os.path.join(data_dir, "augmented")
            os.makedirs(aug_dir, exist_ok=True)

            for i, img in enumerate(aug_imgs):
                fname = f"aug_{len(self.images) + i}.png"
                save_path = os.path.join(aug_dir, fname)
                img.save(save_path)
                self.images.append(save_path)
                self.labels.append(1)  # Always label 1

            print(f"Total training images after augmentation: {len(self.labels)}")

        elif self.mode in ["val", "test"]:
            with open(train_file, "r", encoding="gbk") as file:
                for line in file:
                    fields = [f.strip() for f in line.strip().split(",")]
                    if len(fields) != 3:
                        continue
                    img_name, _, _ = fields
                    path = os.path.join(data_dir, f"{img_name}.jpg")
                    if not os.path.exists(path):
                        path = os.path.join(data_dir, f"{img_name}.png")
                    self.images.append(path)

   
class EdgeDataset(Dataset):
    def __init__(self, data_dir, train_file, mode="train", train_manual=None):
        self.images = []   # str paths or PIL.Image for aug
        self.labels = []
        self.mode = mode

        # 1) load base ground-truth
        if train_manual and mode=="train":
            with open(train_manual, "r", encoding="utf-8") as f:
                for line in f:
                    parts = [x.strip() for x in line.split(",")]
                    if len(parts)!=2: continue
                    name,label = parts
                    p = os.path.join(data_dir, f"{name}.jpg")
                    if not os.path.exists(p): p = os.path.join(data_dir, f"{name}.png")
                    self.images.append(p); self.labels.append(int(label))

        # 2) generate positive collages in‐memory
        if mode=="train" and train_file:
            # gather candidate paths
            paths, labs = [], []
            with open(train_file, "r", encoding="gbk") as f:
                for line in f:
                    parts = [x.strip() for x in line.split(",")]
                    if len(parts)==4:
                        name = parts[0]
                        p = os.path.join(data_dir, f"{name}.jpg")
                        if not os.path.exists(p): p = os.path.join(data_dir, f"{name}.png")
                        paths.append(p); labs.append(int(parts[2]))

            # call each augmenter
            aug1, lbl1 = self.augmentation(paths, labs, augmentation_sample_num=20)
            aug2, lbl2 = self.augmentation_2_2(self.images, self.labels)
            aug3, lbl3 = self.augmentation_1_2(self.images, self.labels)
            aug4, lbl4 = self.augmentation_2h(self.images, self.labels)

            # append all in‐memory PIL.Images
            for img,lb in zip(aug1+aug2+aug3+aug4, lbl1+lbl2+lbl3+lbl4):
                self.images.append(img)
                self.labels.append(lb)

        # 3) val/test just file paths
        elif mode in ("val","test") and train_file:
            with open(train_file, "r", encoding="gbk") as f:
                for line in f:
                    parts = [x.strip() for x in line.split(",")]
                    if len(parts)==3:
                        name=parts[0]
                        p=os.path.join(data_dir,f"{name}.jpg")
                        if not os.path.exists(p): p=os.path.join(data_dir,f"{name}.png")
                        self.images.append(p)


    def augmentation(self, image_paths, labels, augmentation_sample_num=20, unit_size=128):
        # 1) Group file-paths by original label
        grouped = defaultdict(list)
        for p, l in zip(image_paths, labels):
            grouped[l].append(p)

        augmented_images, augmented_labels = [], []

        # 2) For each label‐group, build collages from only that group
        for orig_label, paths in grouped.items():
            pil_imgs = [Image.open(p).convert("RGB") for p in paths]
            random.shuffle(pil_imgs)
            used = 0

            # same loop as before, but only within this group
            while used + 9 <= len(pil_imgs) and len(augmented_images) < augmentation_sample_num:
                n = random.randint(2,4) if random.random()<=0.5 else random.randint(2,9)
                chosen = pil_imgs[used:used+n]
                used += n

                layout = random.choice(("vertical","grid"))
                if layout=="vertical":
                    canvas = Image.new("RGB", (unit_size, unit_size * n))
                    for i, im in enumerate(chosen):
                        canvas.paste(im.resize((unit_size,unit_size)), (0, unit_size*i))
                else:
                    g = int(n**0.5) + (1 if int(n**0.5)**2 < n else 0)
                    canvas = Image.new("RGB", (unit_size*g, unit_size*g))
                    for i, im in enumerate(chosen):
                        r, c = divmod(i, g)
                        canvas.paste(im.resize((unit_size,unit_size)),
                                     (c*unit_size, r*unit_size))

                augmented_images.append(canvas)
                # these are always collages → label them 1
                augmented_labels.append(1)

        return augmented_images, augmented_labels


    def augmentation_2_2(self, images, labels, unit_size=128):
        grouped = defaultdict(list)
        for p, l in zip(images, labels):
            grouped[l].append(p)

        out_i, out_l = [], []
        for orig_label, paths in grouped.items():
            pil_imgs = [Image.open(p).convert("RGB") for p in paths]
            random.shuffle(pil_imgs)
            for i in range(0, (len(pil_imgs)//4)*4, 4):
                canvas = Image.new("RGB", (unit_size*2, unit_size*2))
                for idx in range(4):
                    r, c = divmod(idx, 2)
                    im = pil_imgs[i+idx].resize((unit_size, unit_size))
                    canvas.paste(im, (c*unit_size, r*unit_size))
                out_i.append(canvas); out_l.append(1)
        return out_i, out_l

    # 1 on left + 2 stacked on right, per‐label grouping
    def augmentation_1_2(self, images, labels, unit_size=128):
        grouped = defaultdict(list)
        for p, l in zip(images, labels):
            grouped[l].append(p)

        out_i, out_l = [], []
        for orig_label, paths in grouped.items():
            pil_imgs = [Image.open(p).convert("RGB") for p in paths]
            random.shuffle(pil_imgs)
            used = 0
            while used + 3 <= len(pil_imgs):
                a, b, c = pil_imgs[used:used+3]
                used += 3
                canvas = Image.new("RGB", (unit_size*2, unit_size*2))
                if random.random() < 0.5:  # big on left
                    canvas.paste(a.resize((unit_size, unit_size*2)), (0,0))
                    canvas.paste(b.resize((unit_size, unit_size)),    (unit_size,0))
                    canvas.paste(c.resize((unit_size, unit_size)),    (unit_size,unit_size))
                else:  # big on right
                    canvas.paste(b.resize((unit_size, unit_size)),    (0,0))
                    canvas.paste(c.resize((unit_size, unit_size)),    (0,unit_size))
                    canvas.paste(a.resize((unit_size, unit_size*2)), (unit_size,0))
                out_i.append(canvas); out_l.append(1)
        return out_i, out_l

    # 2 images side by side, per‐label grouping
    def augmentation_2h(self, images, labels, unit_size=128):
        grouped = defaultdict(list)
        for p, l in zip(images, labels):
            grouped[l].append(p)

        out_i, out_l = [], []
        for orig_label, paths in grouped.items():
            pil_imgs = [Image.open(p).convert("RGB") for p in paths]
            random.shuffle(pil_imgs)
            used = 0
            while used + 2 <= len(pil_imgs):
                a, b = pil_imgs[used], pil_imgs[used+1]
                used += 2
                canvas = Image.new("RGB", (unit_size*2, unit_size))
                canvas.paste(a.resize((unit_size, unit_size)), (0,0))
                canvas.paste(b.resize((unit_size, unit_size)), (unit_size,0))
                out_i.append(canvas); out_l.append(1)
        return out_i, out_l

    # 3 images side by side, per‐label grouping
    def augmentation_3h(self, images, labels, unit_size=128):
        grouped = defaultdict(list)
        for p, l in zip(images, labels):
            grouped[l].append(p)

        out_i, out_l = [], []
        for orig_label, paths in grouped.items():
            pil_imgs = [Image.open(p).convert("RGB") for p in paths]
            random.shuffle(pil_imgs)
            used = 0
            while used + 3 <= len(pil_imgs):
                a, b, c = pil_imgs[used:used+3]
                used += 3
                canvas = Image.new("RGB", (unit_size*3, unit_size))
                canvas.paste(a.resize((unit_size, unit_size)), (0,0))
                canvas.paste(b.resize((unit_size, unit_size)), (unit_size,0))
                canvas.paste(c.resize((unit_size, unit_size)), (2*unit_size,0))
                out_i.append(canvas); out_l.append(1)
        return out_i, out_l

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        item = self.images[idx]
        if isinstance(item, str):
            img = Image.open(item).convert("RGB")
        else:
            img = item
        #x = preprocess_edge_map(img)
        if not isinstance(img, torch.Tensor):
            img_resized = img.resize((256, 256))
            # 2) to numpy array (H, W, 3), float32 scaled to [0,1]
            arr = np.array(img_resized, dtype=np.float32) / 255.0
            # 3) to torch tensor and permute to (C, H, W)
            x = torch.from_numpy(arr).permute(2, 0, 1)
        if self.mode=="train":
            return x, torch.tensor(self.labels[idx]).long()
        return x
# ===================== Model =====================
def build_resnet34(in_channels=3):
    model = resnet34(pretrained=ResNet34_Weights.DEFAULT)
    model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Sequential(  # type: ignore
        nn.Dropout(0.5),
        nn.Linear(model.fc.in_features, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(0.5),
        nn.Linear(512, 2),  # <-- single output
    )
    return model

# ===================== Train & Eval =====================
def train(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
    return total_loss / len(loader.dataset)

def compute_threshold_from_train(train_manual, data_dir):
    peaks0, peaks1 = [], []

    with open(train_manual, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or "," not in line:
                continue
            # split on the *last* comma so IDs with commas won't break
            id_str, label_str = line.rsplit(",", 1)
            id_str = id_str.strip()
            try:
                label = int(label_str)
            except ValueError:
                continue

            # locate the image file
            for ext in (".jpg", ".png"):
                path = os.path.join(data_dir, f"{id_str}{ext}")
                if os.path.exists(path):
                    img = Image.open(path)
                    break
            else:
                # no file found
                continue

            # get the raw “peak” (threshold=0 → returns peak)
            _, peak = detect_collage_with_thresh(img, threshold=0)
            if label == 0:
                peaks0.append(peak)
            else:
                peaks1.append(peak)

    # fallback
    if not peaks0 or not peaks1:
        print("Warning: unable to compute both classes—using default threshold=100")
        return 100.0

    # choose data‐driven threshold
    p0 = np.percentile(peaks0, 95)
    p1 = np.percentile(peaks1, 5)
    thresh = (p0 + p1) / 2.0
    print(f"  95%-tile of singles: {p0:.1f}")
    print(f"   5%-tile of collages: {p1:.1f}")
    print(f"→ chosen threshold = {thresh:.1f}")
    return thresh

def evaluate(model, loader, device):
    model.eval()
    preds = []
    with torch.no_grad():
        for x in loader:
            if isinstance(x, (list, tuple)):
                x = x[0]
            x = x.to(device)
            out = model(x)
            probs = torch.softmax(out, dim=1)
            pred = torch.argmax(probs, dim=1).cpu().numpy()
            preds.extend(pred)
    return preds

def evaluate_train(model, loader, device):
    model.eval()
    preds = []
    labels = []
    with torch.no_grad():
        for x, y in loader:
            if isinstance(x, (list, tuple)):
                x = x[0]
            x = x.to(device)
            y = y.to(device)
            out = model(x)
            probs = torch.softmax(out, dim=1)
            pred = torch.argmax(probs, dim=1).cpu().numpy()
            preds.extend(pred)
            labels.extend(y.cpu().numpy())
    return np.array(preds), np.array(labels)

def make_predictions(csv_file, data_dir, output_csv, threshold):
    df = pd.read_csv(csv_file, encoding="gbk", header=None, usecols=[0], names=["id"])
    preds = []
    for img_id in df["id"]:
        # find file
        for ext in (".jpg", ".png"):
            path = os.path.join(data_dir, f"{img_id}{ext}")
            if os.path.exists(path):
                img = Image.open(path)
                break
        else:
            preds.append(0)  # or skip / default
            continue

        pred, _ = detect_collage_with_thresh(img, threshold)
        preds.append(pred)

    return pd.DataFrame(preds)
# ===================== Main =====================
def main(method:str = "CNN"):
    train_dir, train_manual = "data_download/train", "train_manual.csv"
    train_csv = "data/train.csv"
    val_dir, val_csv = "data_download/val", "data/val.csv"
    test_dir, test_csv = "data_download/test", "data/test.csv"

    if method == "CNN":
        train_ds = EdgeDataset(train_dir, train_csv, mode="train", train_manual=train_manual)
        val_ds = EdgeDataset(val_dir, val_csv, mode="val")
        test_ds = EdgeDataset(test_dir, test_csv, mode="test")
        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=2)
        test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=2)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = build_resnet34().to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        for epoch in range(4):
            loss = train(model, train_loader, criterion, optimizer, device)
            print(f"Epoch {epoch+1} - Loss: {loss:.4f}")
            train_preds, train_labels = evaluate_train(model, train_loader, device)

            from sklearn.metrics import accuracy_score, classification_report

            train_acc = accuracy_score(train_labels, train_preds)
            print(f"\nTrain Accuracy: {train_acc:.4f}")
            print("Train Classification Report:")
            print(classification_report(train_labels, train_preds, digits=4))

        val_preds = evaluate(model, val_loader, device)
        test_preds = evaluate(model, test_loader, device)
    elif method == "canny":
        # compute threshold
        threshold = compute_threshold_from_train(train_manual, train_dir)

        # predict & save
        val_preds = make_predictions(val_csv,  val_dir,   "submissionA.csv", threshold)
        test_preds = make_predictions(test_csv, test_dir,  "submissionB.csv", threshold)

    save_csv(val_preds, "submissionA.csv")
    save_csv(test_preds, "submissionB.csv")

    with zipfile.ZipFile("submission.zip", "w") as z:
        z.write("submissionA.csv")
        z.write("submissionB.csv")
    os.remove("submissionA.csv")
    os.remove("submissionB.csv")

if __name__ == "__main__":
    method = "CNN" 
    # method = "canny"
    main(method)
