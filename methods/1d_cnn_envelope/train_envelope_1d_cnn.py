from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score
from torch import nn
from torch.utils.data import DataLoader, Dataset


LABEL_TO_INDEX = {"no_burst": 0, "burst": 1}
INDEX_TO_LABEL = {value: key for key, value in LABEL_TO_INDEX.items()}


def find_project_root(start: Path | None = None) -> Path:
    path = (start or Path.cwd()).resolve()
    for candidate in [path, *path.parents]:
        if (candidate / "dataset" / "metadata.csv").exists():
            return candidate
    raise FileNotFoundError("Could not find project root with dataset/metadata.csv")


class EnvelopeDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, dataset_dir: Path):
        self.frame = frame.reset_index(drop=True)
        self.dataset_dir = dataset_dir

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        row = self.frame.iloc[index]
        npz_path = self.dataset_dir / Path(str(row["npz_path"]).replace("\\", "/"))
        data = np.load(npz_path)
        x = data["envelope_db"].astype(np.float32)
        x = (x - x.mean()) / (x.std() + 1e-6)
        y = np.int64(row["target"])
        return torch.from_numpy(x[None, :]), torch.tensor(y, dtype=torch.long)


class EnvelopeCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=9, padding=4),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(0.2), nn.Linear(64, 2))

    def forward(self, x):
        return self.classifier(self.features(x))


def load_metadata(project_root: Path) -> pd.DataFrame:
    metadata = pd.read_csv(project_root / "dataset" / "metadata.csv")
    metadata = metadata[metadata["label"].isin(LABEL_TO_INDEX)].copy()
    metadata["target"] = metadata["label"].map(LABEL_TO_INDEX).astype(int)
    return metadata


def build_loaders(metadata: pd.DataFrame, dataset_dir: Path, batch_size: int, num_workers: int = 0):
    train_df = metadata[metadata["split"] == "train"].copy()
    val_df = metadata[metadata["split"] == "val"].copy()
    test_df = metadata[metadata["split"] == "test"].copy()
    loaders = {
        "train": DataLoader(EnvelopeDataset(train_df, dataset_dir), batch_size=batch_size, shuffle=True, num_workers=num_workers),
        "val": DataLoader(EnvelopeDataset(val_df, dataset_dir), batch_size=batch_size, shuffle=False, num_workers=num_workers),
        "test": DataLoader(EnvelopeDataset(test_df, dataset_dir), batch_size=batch_size, shuffle=False, num_workers=num_workers),
    }
    frames = {"train": train_df, "val": val_df, "test": test_df}
    return loaders, frames


def run_epoch(model: nn.Module, loader: DataLoader, criterion, optimizer, device: torch.device, train: bool):
    model.train(train)
    total_loss = 0.0
    all_y = []
    all_pred = []
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        with torch.set_grad_enabled(train):
            logits = model(x)
            loss = criterion(logits, y)
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        total_loss += float(loss.item()) * len(y)
        all_y.append(y.detach().cpu().numpy())
        all_pred.append(logits.argmax(dim=1).detach().cpu().numpy())
    y_true = np.concatenate(all_y)
    y_pred = np.concatenate(all_pred)
    return total_loss / len(loader.dataset), f1_score(y_true, y_pred, zero_division=0), y_true, y_pred


def predict_loader(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    ys = []
    preds = []
    probs = []
    with torch.no_grad():
        for x, y in loader:
            logits = model(x.to(device))
            prob = torch.softmax(logits, dim=1).cpu().numpy()
            ys.append(y.numpy())
            preds.append(prob.argmax(axis=1))
            probs.append(prob)
    return np.concatenate(ys), np.concatenate(preds), np.concatenate(probs)


def train(
    epochs: int = 40,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    seed: int = 42,
    output_dir: Path | None = None,
):
    project_root = find_project_root()
    method_dir = Path(__file__).resolve().parent
    output_dir = output_dir or method_dir
    models_dir = output_dir / "models"
    reports_dir = output_dir / "reports"
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(max(1, min(6, torch.get_num_threads())))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    metadata = load_metadata(project_root)
    loaders, frames = build_loaders(metadata, project_root / "dataset", batch_size)
    model = EnvelopeCNN().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    best_state = None
    best_val_f1 = -1.0
    history = []
    for epoch in range(1, epochs + 1):
        train_loss, train_f1, _, _ = run_epoch(model, loaders["train"], criterion, optimizer, device, train=True)
        val_loss, val_f1, _, _ = run_epoch(model, loaders["val"], criterion, optimizer, device, train=False)
        history.append({"epoch": epoch, "train_loss": train_loss, "train_f1": train_f1, "val_loss": val_loss, "val_f1": val_f1})
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        print(f"epoch {epoch:02d}: train_loss={train_loss:.4f} train_f1={train_f1:.4f} val_loss={val_loss:.4f} val_f1={val_f1:.4f}")

    model.load_state_dict(best_state)
    val_y, val_pred, _ = predict_loader(model, loaders["val"], device)
    test_y, test_pred, test_prob = predict_loader(model, loaders["test"], device)

    report = {
        "model_kind": "envelope_1d",
        "train_samples": int(len(frames["train"])),
        "val_samples": int(len(frames["val"])),
        "test_samples": int(len(frames["test"])),
        "best_val_f1": float(best_val_f1),
        "val": {
            "precision": float(precision_score(val_y, val_pred, zero_division=0)),
            "recall": float(recall_score(val_y, val_pred, zero_division=0)),
            "f1": float(f1_score(val_y, val_pred, zero_division=0)),
            "confusion_matrix": confusion_matrix(val_y, val_pred).tolist(),
            "classification_report": classification_report(val_y, val_pred, target_names=[INDEX_TO_LABEL[0], INDEX_TO_LABEL[1]], zero_division=0, output_dict=True),
        },
        "test": {
            "precision": float(precision_score(test_y, test_pred, zero_division=0)),
            "recall": float(recall_score(test_y, test_pred, zero_division=0)),
            "f1": float(f1_score(test_y, test_pred, zero_division=0)),
            "confusion_matrix": confusion_matrix(test_y, test_pred).tolist(),
            "classification_report": classification_report(test_y, test_pred, target_names=[INDEX_TO_LABEL[0], INDEX_TO_LABEL[1]], zero_division=0, output_dict=True),
        },
        "history": history,
    }

    model_path = models_dir / "envelope_1d.pt"
    metrics_path = reports_dir / "envelope_1d_metrics.json"
    predictions_path = reports_dir / "envelope_1d_predictions.csv"

    torch.save({"model_kind": "envelope_1d", "model_state_dict": model.state_dict(), "label_to_index": LABEL_TO_INDEX}, model_path)
    metrics_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    predictions = frames["test"][["sample_id", "source_file", "segment_start", "segment_end", "label", "png_path", "npz_path"]].copy()
    predictions["predicted"] = [INDEX_TO_LABEL[int(value)] for value in test_pred]
    predictions["prob_no_burst"] = test_prob[:, 0]
    predictions["prob_burst"] = test_prob[:, 1]
    predictions.to_csv(predictions_path, index=False)

    print("VAL")
    print(confusion_matrix(val_y, val_pred))
    print(classification_report(val_y, val_pred, target_names=[INDEX_TO_LABEL[0], INDEX_TO_LABEL[1]], zero_division=0))
    print("TEST")
    print(confusion_matrix(test_y, test_pred))
    print(classification_report(test_y, test_pred, target_names=[INDEX_TO_LABEL[0], INDEX_TO_LABEL[1]], zero_division=0))
    print(f"model: {model_path}")
    print(f"metrics: {metrics_path}")
    print(f"predictions: {predictions_path}")

    return model, report, predictions


if __name__ == "__main__":
    train()
