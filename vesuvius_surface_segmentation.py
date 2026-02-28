# ------------------ INSTALL OFFLINE PACKAGES ------------------
# The competition has the rules that the internet must be off so we must
#install all the packages offline.
import os

WHEEL_DIR = "/kaggle/input/notebooks/ipythonx/vsdetection-packages-offline-installer-only/whls"
os.system(f"pip install {WHEEL_DIR}/imagecodecs*.whl --no-index -q")

WHEEL_DIR = "/kaggle/input/datasets/rajwardhandasture/seg-vis-pytorch/smp_wheels"
os.system(f"pip install --no-index --find-links={WHEEL_DIR} segmentation-models-pytorch -q")


# ------------------ IMPORTS ------------------
import numpy as np
import torch
import torch.nn as nn
import tifffile
import zipfile
import segmentation_models_pytorch as smp
import pandas as pd

from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import f1_score


# ------------------ CONFIG ------------------
DEVICE  = "cuda" if torch.cuda.is_available() else "cpu"

DEPTH   = 16      # number of z-slices used as channels
PATCH   = 128     # tile size
STRIDE  = 112
BATCH   = 4
EPOCHS  = 9
LR      = 1e-4

TRAIN_DIR     = "/kaggle/input/competitions/vesuvius-challenge-surface-detection/train_images"
LABEL_DIR     = "/kaggle/input/competitions/vesuvius-challenge-surface-detection/train_labels"
DEP_TRAIN_DIR = "/kaggle/input/competitions/vesuvius-challenge-surface-detection/deprecated_train_images"
DEP_LABEL_DIR = "/kaggle/input/competitions/vesuvius-challenge-surface-detection/deprecated_train_labels"
TEST_DIR      = "/kaggle/input/competitions/vesuvius-challenge-surface-detection/test_images"

WEIGHTS_PATH  = "/kaggle/input/models/rajwardhandasture/resnet-weights/pytorch/default/1/resnet34.pth"
CKPT_PATH     = "/kaggle/working/best_model.pth"


# ------------------ HELPERS ------------------
def normalize(vol):
    """Percentile normalization."""
    vol = vol.astype(np.float32)
    lo, hi = np.percentile(vol, (1, 99))
    vol = np.clip(vol, lo, hi)
    return (vol - lo) / (hi - lo + 1e-6)


def get_positions(size, patch, stride):
    """Sliding window coordinates."""
    pos = list(range(0, size - patch + 1, stride))
    if not pos or pos[-1] != size - patch:
        pos.append(size - patch)
    return pos


# ------------------ DATASET ------------------
class ScrollDataset(Dataset):
    """Loads volume once and yields training patches."""

    def __init__(self, vol_path, label_path, patch, stride, depth):
        self.vol_path = vol_path
        self.label_path = label_path
        self.patch = patch
        self.depth = depth
        self.half = depth // 2

        tmp = tifffile.imread(vol_path)
        Z, H, W = tmp.shape
        del tmp

        ys = get_positions(H, patch, stride)
        xs = get_positions(W, patch, stride)
        zs = list(range(self.half, Z - self.half))
        self.tiles = [(z, y, x) for z in zs for y in ys for x in xs]

        self.vol   = normalize(tifffile.imread(vol_path))
        self.label = tifffile.imread(label_path).astype(np.int64)

    def __len__(self):
        return len(self.tiles)

    def __getitem__(self, idx):
        z, y, x = self.tiles[idx]
        p, h = self.patch, self.half

        stack = self.vol[z-h:z+h, y:y+p, x:x+p].copy()
        mask  = self.label[z, y:y+p, x:x+p].copy()

        # simple flips for augmentation
        if np.random.rand() > 0.5:
            stack = stack[:, ::-1, :]
            mask  = mask[::-1, :]
        if np.random.rand() > 0.5:
            stack = stack[:, :, ::-1]
            mask  = mask[:, ::-1]

        return (
            torch.tensor(stack, dtype=torch.float32),
            torch.tensor(mask, dtype=torch.long),
        )


# ------------------ MODEL ------------------
def build_model():
    """UNet with pretrained ResNet34 encoder adapted to 16 channels."""

    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=DEPTH,
        classes=3,
    ).to(DEVICE)

    state_dict = torch.load(WEIGHTS_PATH, map_location="cpu")
    old_weight = state_dict["conv1.weight"]

    avg_weight = old_weight.mean(dim=1, keepdim=True)
    new_weight = avg_weight.repeat(1, DEPTH, 1, 1)

    state_dict["conv1.weight"] = new_weight
    state_dict.pop("fc.weight", None)
    state_dict.pop("fc.bias", None)

    model.encoder.load_state_dict(state_dict, strict=False)
    return model


# ------------------ LOSS ------------------
class_weights = torch.tensor([1.0, 15.0, 1.0]).to(DEVICE)
criterion = nn.CrossEntropyLoss(weight=class_weights)


# ------------------ TRAINING ------------------
def train():

    main_ids = [(TRAIN_DIR, LABEL_DIR, f)
                for f in sorted(os.listdir(TRAIN_DIR)) if f.endswith(".tif")][:30]

    dep_ids = [(DEP_TRAIN_DIR, DEP_LABEL_DIR, f)
               for f in sorted(os.listdir(DEP_TRAIN_DIR))
               if f.endswith(".tif") and os.path.exists(f"{DEP_LABEL_DIR}/{f}")]

    train_ids = main_ids + dep_ids

    model = build_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    scaler = torch.amp.GradScaler('cuda')

    best_val_loss = float("inf")

    for epoch in range(EPOCHS):

        train_loss = 0
        val_loss = 0
        n_train = 0
        n_val = 0

        for vol_dir, lbl_dir, fname in train_ids:

            vol_path = f"{vol_dir}/{fname}"
            label_path = f"{lbl_dir}/{fname}"
            if not os.path.exists(label_path):
                continue

            ds = ScrollDataset(vol_path, label_path, PATCH, STRIDE, DEPTH)

            nv = max(1, int(0.1 * len(ds)))
            nt = len(ds) - nv
            train_ds, val_ds = random_split(ds, [nt, nv])

            train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True)
            val_loader   = DataLoader(val_ds, batch_size=BATCH)

            model.train()
            for imgs, masks in train_loader:
                imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)

                optimizer.zero_grad()
                with torch.amp.autocast('cuda'):
                    loss = criterion(model(imgs), masks)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                train_loss += loss.item()
                n_train += 1

            model.eval()
            with torch.no_grad():
                for imgs, masks in val_loader:
                    imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
                    with torch.amp.autocast('cuda'):
                        loss = criterion(model(imgs), masks)
                    val_loss += loss.item()
                    n_val += 1

            del ds
            torch.cuda.empty_cache()

        train_loss /= max(n_train, 1)
        val_loss   /= max(n_val, 1)
        scheduler.step()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), CKPT_PATH)


# ------------------ SLIDING WINDOW INFERENCE ------------------
def sliding_window_predict_probs(model, volume):

    Z, H, W = volume.shape
    half = DEPTH // 2

    pred_vol  = np.zeros((Z, H, W), np.float32)
    count_vol = np.zeros((Z, H, W), np.float32)

    ys = get_positions(H, PATCH, STRIDE)
    xs = get_positions(W, PATCH, STRIDE)

    model.eval()

    for z in range(half, Z - half):
        stack = volume[z-half:z+half]

        for y in ys:
            for x in xs:
                patch_stack = stack[:, y:y+PATCH, x:x+PATCH]
                inp = torch.tensor(patch_stack).unsqueeze(0).to(DEVICE)

                with torch.no_grad():
                    with torch.amp.autocast('cuda'):
                        logits = model(inp)
                        probs = torch.softmax(logits, dim=1)
                        surface_prob = probs[0,1].cpu().numpy()

                pred_vol[z, y:y+PATCH, x:x+PATCH] += surface_prob
                count_vol[z, y:y+PATCH, x:x+PATCH] += 1

    pred_vol /= np.maximum(count_vol, 1e-6)
    return pred_vol


# ------------------ TRAIN MODEL ------------------
if os.path.exists(CKPT_PATH):
    os.remove(CKPT_PATH)

train()


# ------------------ LOAD BEST MODEL ------------------
model = build_model()
model.load_state_dict(torch.load(CKPT_PATH, map_location=DEVICE))
model.eval()


# ------------------ THRESHOLD SEARCH ------------------
dep_label = tifffile.imread(
    "/kaggle/input/competitions/vesuvius-challenge-surface-detection/deprecated_train_labels/1407735.tif"
)

surface_gt = (dep_label == 1).astype(np.uint8)

vol = normalize(tifffile.imread(f"{TEST_DIR}/1407735.tif"))
prob_vol = sliding_window_predict_probs(model, vol)

best_thresh, best_f1 = 0.35, 0.0

for thresh in np.arange(0.1, 0.7, 0.05):
    pred = (prob_vol > thresh).astype(np.uint8)
    f1 = f1_score(surface_gt.flatten(), pred.flatten())
    if f1 > best_f1:
        best_f1, best_thresh = f1, thresh


# ------------------ CREATE SUBMISSION ------------------
OUT_DIR = "/kaggle/working/masks"
ZIP_PATH = "/kaggle/working/submission.zip"
os.makedirs(OUT_DIR, exist_ok=True)

test_csv = pd.read_csv(
    "/kaggle/input/competitions/vesuvius-challenge-surface-detection/test.csv"
)

with zipfile.ZipFile(ZIP_PATH, "w") as z:
    for id_str in test_csv["id"].astype(str):

        fname = f"{id_str}.tif"
        vol_path = f"{TEST_DIR}/{fname}"
        if not os.path.exists(vol_path):
            continue

        vol = normalize(tifffile.imread(vol_path))
        prob_vol = sliding_window_predict_probs(model, vol)

        pred = (prob_vol > best_thresh).astype(np.uint8)

        out_path = f"{OUT_DIR}/{fname}"
        tifffile.imwrite(out_path, pred, compression=None)
        z.write(out_path, arcname=fname)