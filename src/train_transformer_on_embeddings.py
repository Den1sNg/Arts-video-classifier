import copy
import json
import random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
SEED = 42
MANIFEST_CSV = Path("/content/embeddings_manifest.csv")
EMBEDDINGS_DIR = Path("/content/embeddings")
MODEL_DIR = Path("/content/models")
RESULTS_DIR = Path("/content/model_results")
SPLITS_DIR = Path("/content/splits")
BATCH_SIZE = 32
EPOCHS = 25
LR = 1e-4
WEIGHT_DECAY = 1e-4
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
def normalize_label(label):
    return str(label).strip().lower().replace(" ", "_")
def build_embedding_index():
    index = {}
    for path in EMBEDDINGS_DIR.rglob("*.pt"):
        index[path.stem] = path
    return index
def resolve_embedding_path(row, embedding_index):
    possible_cols = ["embedding_path", "path", "file_path", "pt_path", "embedding_file"]
    for col in possible_cols:
        if col in row.index and pd.notna(row[col]):
            p = Path(str(row[col]))
            if p.exists():
                return p
            candidate = Path("/content") / p.name
            if candidate.exists():
                return candidate
            if p.name in embedding_index:
                return embedding_index[p.name]
    video_id = str(row["video_id"])
    label = normalize_label(row["label"])
    candidates = [
        EMBEDDINGS_DIR / label / f"{video_id}.pt",
        EMBEDDINGS_DIR / f"{video_id}.pt",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return embedding_index.get(video_id)
def load_valid_dataframe():
    if not MANIFEST_CSV.exists():
        raise FileNotFoundError(f"Manifest not found: {MANIFEST_CSV}")
    if not EMBEDDINGS_DIR.exists():
        raise FileNotFoundError(f"Embeddings directory not found: {EMBEDDINGS_DIR}")
    df = pd.read_csv(MANIFEST_CSV)
    if "status" in df.columns:
        df = df[df["status"] == "ok"].copy()
    if "label" not in df.columns:
        raise ValueError("Manifest must contain column: label")
    if "video_id" not in df.columns:
        raise ValueError("Manifest must contain column: video_id")
    df["label"] = df["label"].apply(normalize_label)
    embedding_index = build_embedding_index()
    resolved_paths = []
    for _, row in df.iterrows():
        path = resolve_embedding_path(row, embedding_index)
        resolved_paths.append(str(path) if path is not None else None)
    df["resolved_path"] = resolved_paths
    df = df[df["resolved_path"].notna()].copy()
    df = df.drop_duplicates(subset=["video_id"]).reset_index(drop=True)
    return df
def make_splits(df):
    counts = df["label"].value_counts()
    valid_labels = counts[counts >= 3].index.tolist()
    removed_labels = counts[counts < 3]
    if len(removed_labels) > 0:
        print("Removed labels with less than 3 examples:")
        print(removed_labels)
    df = df[df["label"].isin(valid_labels)].reset_index(drop=True)
    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        random_state=SEED,
        stratify=df["label"],
    )
    temp_counts = temp_df["label"].value_counts()
    if temp_counts.min() >= 2:
        val_df, test_df = train_test_split(
            temp_df,
            test_size=0.50,
            random_state=SEED,
            stratify=temp_df["label"],
        )
    else:
        val_df, test_df = train_test_split(
            temp_df,
            test_size=0.50,
            random_state=SEED,
        )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)
class EmbeddingDataset(Dataset):
    def __init__(self, df, label_to_id):
        self.df = df.reset_index(drop=True)
        self.label_to_id = label_to_id
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        emb = torch.load(row["resolved_path"], map_location="cpu")
        if isinstance(emb, dict):
            for key in ["embedding", "embeddings", "features", "x"]:
                if key in emb:
                    emb = emb[key]
                    break
        emb = torch.as_tensor(emb, dtype=torch.float32)
        if emb.ndim == 3:
            emb = emb.squeeze(0)
        if emb.ndim != 2:
            raise ValueError(f"Expected embedding shape [frames, dim], got {tuple(emb.shape)}")
        y = self.label_to_id[row["label"]]
        return emb, torch.tensor(y, dtype=torch.long)
class VideoTransformerClassifier(nn.Module):
    def __init__(
        self,
        input_dim,
        num_classes,
        max_len,
        d_model=256,
        nhead=8,
        num_layers=3,
        dropout=0.2,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )
    def forward(self, x):
        x = self.input_proj(x)
        x = x + self.pos_embedding[:, : x.size(1), :]
        x = self.encoder(x)
        x = self.norm(x)
        x = x.mean(dim=1)
        return self.classifier(x)
def evaluate(model, loader, device):
    model.eval()
    preds = []
    targets = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            pred = logits.argmax(dim=1).cpu().numpy().tolist()
            preds.extend(pred)
            targets.extend(y.numpy().tolist())
    acc = accuracy_score(targets, preds)
    f1 = f1_score(targets, preds, average="macro", zero_division=0)
    return acc, f1, targets, preds
def main():
    set_seed(SEED)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)
    df = load_valid_dataframe()
    print("Total valid embeddings:", len(df))
    print("Labels:")
    print(df["label"].value_counts())
    train_df, val_df, test_df = make_splits(df)
    print()
    print("Split sizes:")
    print("Train:", len(train_df))
    print("Val:", len(val_df))
    print("Test:", len(test_df))
    print("Total used:", len(train_df) + len(val_df) + len(test_df))
    train_df.to_csv(SPLITS_DIR / "train.csv", index=False)
    val_df.to_csv(SPLITS_DIR / "val.csv", index=False)
    test_df.to_csv(SPLITS_DIR / "test.csv", index=False)
    train_df.to_csv(RESULTS_DIR / "train_split.csv", index=False)
    val_df.to_csv(RESULTS_DIR / "val_split.csv", index=False)
    test_df.to_csv(RESULTS_DIR / "test_split.csv", index=False)
    labels = sorted(train_df["label"].unique())
    label_to_id = {label: i for i, label in enumerate(labels)}
    id_to_label = {i: label for label, i in label_to_id.items()}
    with open(RESULTS_DIR / "labels.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "label_to_id": label_to_id,
                "id_to_label": id_to_label,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    train_ds = EmbeddingDataset(train_df, label_to_id)
    val_ds = EmbeddingDataset(val_df, label_to_id)
    test_ds = EmbeddingDataset(test_df, label_to_id)
    sample_x, _ = train_ds[0]
    input_dim = sample_x.shape[-1]
    max_len = sample_x.shape[0]
    num_classes = len(labels)
    print()
    print("Embedding shape:", tuple(sample_x.shape))
    print("Classes:", label_to_id)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
    class_counts = train_df["label"].value_counts()
    class_weights = []
    for label in labels:
        weight = len(train_df) / (num_classes * class_counts[label])
        class_weights.append(weight)
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
    model = VideoTransformerClassifier(
        input_dim=input_dim,
        num_classes=num_classes,
        max_len=max_len,
    ).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    best_val_f1 = -1.0
    best_state_dict = None
    history = []
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}"):
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.size(0)
        train_loss = total_loss / len(train_ds)
        val_acc, val_f1, _, _ = evaluate(model, val_loader, device)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_accuracy": val_acc,
            "val_f1_macro": val_f1,
        }
        history.append(row)
        print(
            f"Epoch {epoch}: "
            f"train_loss={train_loss:.4f}, "
            f"val_acc={val_acc:.4f}, "
            f"val_f1_macro={val_f1:.4f}"
        )
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state_dict = copy.deepcopy(model.state_dict())
            checkpoint = {
                "model_state_dict": best_state_dict,
                "label_to_id": label_to_id,
                "id_to_label": id_to_label,
                "input_dim": input_dim,
                "max_len": max_len,
                "num_classes": num_classes,
                "best_val_f1_macro": best_val_f1,
            }
            torch.save(checkpoint, MODEL_DIR / "best_model.pt")
            torch.save(checkpoint, RESULTS_DIR / "best_transformer.pt")
    pd.DataFrame(history).to_csv(RESULTS_DIR / "history.csv", index=False)
    pd.DataFrame(history).to_csv(RESULTS_DIR / "training_history.csv", index=False)
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
    test_acc, test_f1, y_true, y_pred = evaluate(model, test_loader, device)
    target_names = [id_to_label[i] for i in range(num_classes)]
    report = classification_report(
        y_true,
        y_pred,
        target_names=target_names,
        zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    cm_df = pd.DataFrame(cm, index=target_names, columns=target_names)
    results = {
        "test_accuracy": test_acc,
        "test_f1_macro": test_f1,
        "best_val_f1_macro": best_val_f1,
        "total_valid_embeddings": int(len(df)),
        "total_used": int(len(train_df) + len(val_df) + len(test_df)),
        "train_size": int(len(train_df)),
        "val_size": int(len(val_df)),
        "test_size": int(len(test_df)),
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "label_to_id": label_to_id,
        "history": history,
    }
    with open(RESULTS_DIR / "results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    with open(RESULTS_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    with open(RESULTS_DIR / "classification_report.txt", "w", encoding="utf-8") as f:
        f.write(report)
    cm_df.to_csv(RESULTS_DIR / "confusion_matrix.csv")
    print()
    print("Test accuracy:", test_acc)
    print("Test F1 macro:", test_f1)
    print()
    print(report)
    print("Saved model:", MODEL_DIR / "best_model.pt")
    print("Saved results:", RESULTS_DIR)
if __name__ == "__main__":
    main()
