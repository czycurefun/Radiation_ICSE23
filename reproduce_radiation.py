from __future__ import annotations

import argparse
import csv
import json
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parent
os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / ".hf_cache"))
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup


@dataclass
class Metrics:
    loss: float
    accuracy: float
    precision: float
    recall: float
    f1: float
    tp: int
    fp: int
    fn: int
    total: int


@dataclass
class PairExample:
    left: str
    right: str
    label: int


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_pairs(path: Path) -> list[PairExample]:
    examples: list[PairExample] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parts = line.split("^^")
        if len(parts) != 3:
            continue
        examples.append(PairExample(left=parts[0].strip(), right=parts[1].strip(), label=int(parts[2])))
    return examples


def limit_balanced(examples: list[PairExample], limit: int, seed: int) -> list[PairExample]:
    if limit <= 0 or len(examples) <= limit:
        return examples
    rng = random.Random(seed)
    positives = [item for item in examples if item.label == 1]
    negatives = [item for item in examples if item.label == 0]
    rng.shuffle(positives)
    rng.shuffle(negatives)
    per_class = max(limit // 2, 1)
    selected = positives[:per_class] + negatives[: limit - per_class]
    if len(selected) < limit:
        used = set(id(item) for item in selected)
        rest = [item for item in examples if id(item) not in used]
        rng.shuffle(rest)
        selected.extend(rest[: limit - len(selected)])
    rng.shuffle(selected)
    return selected


class PairDataset(Dataset):
    def __init__(self, examples: list[PairExample], tokenizer: AutoTokenizer, max_length: int) -> None:
        self.examples = examples
        self.encodings = tokenizer(
            [item.left for item in examples],
            [item.right for item in examples],
            max_length=max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        self.labels = torch.tensor([item.label for item in examples], dtype=torch.long)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        item = {key: value[index] for key, value in self.encodings.items()}
        item["labels"] = self.labels[index]
        return item


class BertEncoder(nn.Module):
    def __init__(self, model_name: str) -> None:
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name)
        self.hidden_size = int(self.model.config.hidden_size)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, token_type_ids: torch.Tensor | None = None) -> torch.Tensor:
        kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None:
            kwargs["token_type_ids"] = token_type_ids
        outputs = self.model(**kwargs)
        pooled = getattr(outputs, "pooler_output", None)
        if pooled is None:
            pooled = outputs.last_hidden_state[:, 0]
        return pooled


class LinkClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 96) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LeakyReLU(inplace=False),
            nn.Dropout(0.1),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


class Discriminator(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 96) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def batch_features(encoder: BertEncoder, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    return encoder(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        token_type_ids=batch.get("token_type_ids"),
    )


def gaussian_kernel(source: torch.Tensor, target: torch.Tensor, kernel_mul: float = 2.0, kernel_num: int = 5) -> torch.Tensor:
    n_samples = int(source.size(0)) + int(target.size(0))
    total = torch.cat([source, target], dim=0)
    total0 = total.unsqueeze(0).expand(total.size(0), total.size(0), total.size(1))
    total1 = total.unsqueeze(1).expand(total.size(0), total.size(0), total.size(1))
    l2_distance = ((total0 - total1) ** 2).sum(2)
    denom = max(n_samples**2 - n_samples, 1)
    bandwidth = torch.sum(l2_distance.detach()) / denom
    bandwidth = torch.clamp(bandwidth, min=1e-6)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
    return sum(torch.exp(-l2_distance / item) for item in bandwidth_list)


def mmd(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    batch_size = min(int(source.size(0)), int(target.size(0)))
    source = source[:batch_size]
    target = target[:batch_size]
    kernels = gaussian_kernel(source, target)
    xx = kernels[:batch_size, :batch_size]
    yy = kernels[batch_size:, batch_size:]
    xy = kernels[:batch_size, batch_size:]
    yx = kernels[batch_size:, :batch_size]
    return torch.mean(xx + yy - xy - yx)


def compute_metrics(loss_sum: float, labels: list[int], preds: list[int]) -> Metrics:
    labels_array = np.array(labels)
    preds_array = np.array(preds)
    total = int(len(labels))
    tp = int(((preds_array == 1) & (labels_array == 1)).sum())
    fp = int(((preds_array == 1) & (labels_array == 0)).sum())
    fn = int(((preds_array == 0) & (labels_array == 1)).sum())
    accuracy = float((preds_array == labels_array).mean()) if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return Metrics(
        loss=float(loss_sum / max(total, 1)),
        accuracy=accuracy,
        precision=float(precision),
        recall=float(recall),
        f1=float(f1),
        tp=tp,
        fp=fp,
        fn=fn,
        total=total,
    )


@torch.no_grad()
def evaluate(
    encoder: BertEncoder,
    classifier: LinkClassifier,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
) -> tuple[Metrics, list[int]]:
    encoder.eval()
    classifier.eval()
    labels: list[int] = []
    preds: list[int] = []
    loss_sum = 0.0
    for batch in loader:
        batch = move_batch(batch, device)
        features = batch_features(encoder, batch)
        logits = classifier(features)
        loss = criterion(logits, batch["labels"])
        pred = logits.argmax(dim=1)
        loss_sum += float(loss.item()) * int(batch["labels"].numel())
        labels.extend(batch["labels"].detach().cpu().tolist())
        preds.extend(pred.detach().cpu().tolist())
    return compute_metrics(loss_sum, labels, preds), preds


def cycle_loader(loader: DataLoader) -> Iterable[dict[str, torch.Tensor]]:
    while True:
        for batch in loader:
            yield batch


def train_source(
    encoder: BertEncoder,
    classifier: LinkClassifier,
    source_loader: DataLoader,
    target_loader: DataLoader,
    eval_loader: DataLoader,
    args: argparse.Namespace,
    device: torch.device,
) -> Metrics:
    encoder.train()
    classifier.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(classifier.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    total_steps = max(len(source_loader) * args.source_epochs, 1)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(int(0.1 * total_steps), 0),
        num_training_steps=total_steps,
    )
    target_iter = cycle_loader(target_loader)
    for epoch in range(args.source_epochs):
        for step, source_batch in enumerate(source_loader, start=1):
            target_batch = next(target_iter)
            source_batch = move_batch(source_batch, device)
            target_batch = move_batch(target_batch, device)
            optimizer.zero_grad(set_to_none=True)
            source_features = batch_features(encoder, source_batch)
            target_features = batch_features(encoder, target_batch)
            logits = classifier(source_features)
            cls_loss = criterion(logits, source_batch["labels"])
            distance_loss = mmd(source_features, target_features)
            loss = cls_loss + args.mmd_weight * distance_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(encoder.parameters()) + list(classifier.parameters()), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            if step % args.log_every == 0:
                print(
                    f"[source] epoch={epoch + 1}/{args.source_epochs} "
                    f"step={step}/{len(source_loader)} loss={loss.item():.4f} "
                    f"cls={cls_loss.item():.4f} mmd={distance_loss.item():.4f}"
                )
    metrics, _ = evaluate(encoder, classifier, eval_loader, device, criterion)
    return metrics


def train_adaptation(
    source_encoder: BertEncoder,
    target_encoder: BertEncoder,
    discriminator: Discriminator,
    source_loader: DataLoader,
    target_loader: DataLoader,
    args: argparse.Namespace,
    device: torch.device,
) -> None:
    source_encoder.eval()
    target_encoder.train()
    discriminator.train()
    criterion = nn.CrossEntropyLoss()
    optimizer_d = torch.optim.AdamW(discriminator.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    optimizer_t = torch.optim.AdamW(target_encoder.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    len_loader = min(len(source_loader), len(target_loader))
    for epoch in range(args.adapt_epochs):
        for step, (source_batch, target_batch) in enumerate(zip(source_loader, target_loader), start=1):
            source_batch = move_batch(source_batch, device)
            target_batch = move_batch(target_batch, device)

            with torch.no_grad():
                source_features = batch_features(source_encoder, source_batch)
            target_features = batch_features(target_encoder, target_batch)

            optimizer_d.zero_grad(set_to_none=True)
            concat_features = torch.cat([source_features, target_features.detach()], dim=0)
            domain_logits = discriminator(concat_features)
            domain_labels = torch.cat(
                [
                    torch.ones(source_features.size(0), dtype=torch.long, device=device),
                    torch.zeros(target_features.size(0), dtype=torch.long, device=device),
                ],
                dim=0,
            )
            d_loss = criterion(domain_logits, domain_labels)
            d_loss.backward()
            optimizer_d.step()

            optimizer_t.zero_grad(set_to_none=True)
            target_features = batch_features(target_encoder, target_batch)
            fool_logits = discriminator(target_features)
            fool_labels = torch.ones(target_features.size(0), dtype=torch.long, device=device)
            adv_loss = criterion(fool_logits, fool_labels)
            distance_loss = mmd(source_features.detach(), target_features)
            t_loss = adv_loss + args.mmd_weight * distance_loss
            t_loss.backward()
            torch.nn.utils.clip_grad_norm_(target_encoder.parameters(), args.max_grad_norm)
            optimizer_t.step()

            pred_domain = domain_logits.argmax(dim=1)
            domain_acc = float((pred_domain == domain_labels).float().mean().item())
            if step % args.log_every == 0:
                print(
                    f"[adapt] epoch={epoch + 1}/{args.adapt_epochs} "
                    f"step={step}/{len_loader} d_loss={d_loss.item():.4f} "
                    f"target_loss={t_loss.item():.4f} adv={adv_loss.item():.4f} "
                    f"mmd={distance_loss.item():.4f} d_acc={domain_acc:.4f}"
                )


def save_predictions(
    examples: list[PairExample],
    preds: list[int],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file, delimiter="\t")
        writer.writerow(["left", "right", "label", "prediction"])
        for example, pred in zip(examples, preds):
            writer.writerow([example.left, example.right, example.label, pred])


def save_models(
    output_dir: Path,
    source_encoder: BertEncoder,
    target_encoder: BertEncoder,
    classifier: LinkClassifier,
    discriminator: Discriminator,
) -> None:
    model_dir = output_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    torch.save(source_encoder.state_dict(), model_dir / "source_encoder.pt")
    torch.save(target_encoder.state_dict(), model_dir / "target_encoder.pt")
    torch.save(classifier.state_dict(), model_dir / "classifier.pt")
    torch.save(discriminator.state_dict(), model_dir / "discriminator.pt")


def build_loaders(args: argparse.Namespace, tokenizer: AutoTokenizer) -> tuple[dict[str, list[PairExample]], dict[str, DataLoader]]:
    data_root = PROJECT_ROOT / "data" / "processed"
    source_train = limit_balanced(read_pairs(data_root / args.source / "newtrain.txt"), args.source_train_limit, args.seed)
    source_test = limit_balanced(read_pairs(data_root / args.source / "newtest.txt"), args.eval_limit, args.seed)
    target_train = limit_balanced(read_pairs(data_root / args.target / "newtrain.txt"), args.target_train_limit, args.seed)
    target_test = limit_balanced(read_pairs(data_root / args.target / "newtest.txt"), args.eval_limit, args.seed)
    datasets = {
        "source_train": source_train,
        "source_test": source_test,
        "target_train": target_train,
        "target_test": target_test,
    }
    loaders = {
        name: DataLoader(
            PairDataset(items, tokenizer, args.max_length),
            batch_size=args.batch_size,
            shuffle=name.endswith("train"),
        )
        for name, items in datasets.items()
    }
    return datasets, loaders


def run(args: argparse.Namespace) -> int:
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if args.device != "auto" else ("cuda:0" if torch.cuda.is_available() else "cpu"))
    print(f"device={device}")
    print(f"model={args.model_name}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    datasets, loaders = build_loaders(args, tokenizer)
    for name, items in datasets.items():
        positives = sum(1 for item in items if item.label == 1)
        print(f"{name}: {len(items)} examples, positive={positives}, negative={len(items) - positives}")

    source_encoder = BertEncoder(args.model_name).to(device)
    classifier = LinkClassifier(source_encoder.hidden_size, args.hidden_dim).to(device)
    target_encoder = BertEncoder(args.model_name).to(device)
    target_encoder.load_state_dict(source_encoder.state_dict())
    discriminator = Discriminator(source_encoder.hidden_size, args.hidden_dim).to(device)

    criterion = nn.CrossEntropyLoss()
    source_metrics = train_source(
        source_encoder,
        classifier,
        loaders["source_train"],
        loaders["target_train"],
        loaders["source_test"],
        args,
        device,
    )
    source_only_target_metrics, source_only_preds = evaluate(
        source_encoder, classifier, loaders["target_test"], device, criterion
    )

    train_adaptation(
        source_encoder,
        target_encoder,
        discriminator,
        loaders["source_train"],
        loaders["target_train"],
        args,
        device,
    )
    adapted_target_metrics, adapted_preds = evaluate(
        target_encoder, classifier, loaders["target_test"], device, criterion
    )

    save_models(output_dir, source_encoder, target_encoder, classifier, discriminator)
    save_predictions(datasets["target_test"], source_only_preds, output_dir / "source_only_target_predictions.tsv")
    save_predictions(datasets["target_test"], adapted_preds, output_dir / "adapted_target_predictions.tsv")

    result = {
        "method": "RADIATION reproduction: source requirement-linking classifier plus distance-enhanced adversarial domain adaptation",
        "config": {
            "source": args.source,
            "target": args.target,
            "model_name": args.model_name,
            "device": str(device),
            "batch_size": args.batch_size,
            "max_length": args.max_length,
            "source_epochs": args.source_epochs,
            "adapt_epochs": args.adapt_epochs,
            "source_train_limit": args.source_train_limit,
            "target_train_limit": args.target_train_limit,
            "eval_limit": args.eval_limit,
            "mmd_weight": args.mmd_weight,
            "seed": args.seed,
        },
        "dataset_sizes": {name: len(items) for name, items in datasets.items()},
        "metrics": {
            "source_domain_eval": asdict(source_metrics),
            "source_only_target_eval": asdict(source_only_target_metrics),
            "adapted_target_eval": asdict(adapted_target_metrics),
        },
        "outputs": {
            "models": str(output_dir / "models"),
            "source_only_predictions": str(output_dir / "source_only_target_predictions.tsv"),
            "adapted_predictions": str(output_dir / "adapted_target_predictions.tsv"),
        },
    }
    (output_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result["metrics"], ensure_ascii=False, indent=2))
    print(f"saved: {output_dir / 'result.json'}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a compact RADIATION reproduction.")
    parser.add_argument("--source", default="easy", choices=["easy", "Infusion", "CM1", "EBT", "hippa"])
    parser.add_argument("--target", default="EBT", choices=["easy", "Infusion", "CM1", "EBT", "hippa"])
    parser.add_argument("--model-name", default="prajjwal1/bert-tiny")
    parser.add_argument("--output-dir", default="outputs/reproduction_easy_to_ebt")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--source-epochs", type=int, default=1)
    parser.add_argument("--adapt-epochs", type=int, default=1)
    parser.add_argument("--source-train-limit", type=int, default=64)
    parser.add_argument("--target-train-limit", type=int, default=64)
    parser.add_argument("--eval-limit", type=int, default=0)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--mmd-weight", type=float, default=0.1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--seed", type=int, default=2026)
    return parser


if __name__ == "__main__":
    raise SystemExit(run(build_parser().parse_args()))
