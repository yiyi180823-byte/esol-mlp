"""Train and evaluate a reproducible PyTorch MLP on the Delaney ESOL data."""

from __future__ import annotations

import copy
import json
import logging
import os
import random
from pathlib import Path
from typing import Any

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
OUTPUT_DIR = PROJECT_DIR / "outputs"
FIGURE_DIR = PROJECT_DIR / "figures"
os.environ.setdefault("DEEPCHEM_DATA_DIR", str(DATA_DIR))

logging.disable(logging.WARNING)
import deepchem as dc
logging.disable(logging.NOTSET)
import matplotlib
import numpy as np
import pandas as pd
import rdkit
import sklearn
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from rdkit import RDLogger
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model import ESOLMLP


SEED = 42
INPUT_DIM = 1024
HIDDEN_DIM_1 = 256
HIDDEN_DIM_2 = 64
DROPOUT = 0.2
LEARNING_RATE = 0.001
BATCH_SIZE = 32
MAX_EPOCHS = 100
PATIENCE = 10
DEVICE = torch.device("cpu")
RDLogger.DisableLog("rdApp.warning")


def set_reproducible_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def load_esol() -> tuple[Any, Any, Any]:
    """Load the fixed ECFP/scaffold/no-transformer ESOL configuration."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    _, datasets, _ = dc.molnet.load_delaney(
        featurizer="ECFP",
        splitter="scaffold",
        transformers=[],
    )
    train_dataset, valid_dataset, test_dataset = datasets

    all_features = np.concatenate(
        [train_dataset.X, valid_dataset.X, test_dataset.X], axis=0
    )
    all_targets = np.concatenate(
        [train_dataset.y, valid_dataset.y, test_dataset.y], axis=0
    ).reshape(-1)

    if len(all_targets) != 1128:
        raise RuntimeError(f"Expected 1128 ESOL samples, found {len(all_targets)}.")
    if all_features.ndim != 2 or all_features.shape[1] != INPUT_DIM:
        raise RuntimeError(
            f"Expected an (n, {INPUT_DIM}) feature matrix, found {all_features.shape}."
        )
    if not np.isfinite(all_features).all() or not np.isfinite(all_targets).all():
        raise RuntimeError("Features or targets contain missing/non-finite values.")

    print(
        "Data check passed: "
        f"n={len(all_targets)}, X={all_features.shape}, "
        f"logS mean={all_targets.mean():.4f}, std={all_targets.std():.4f}, "
        f"range=[{all_targets.min():.4f}, {all_targets.max():.4f}]"
    )
    print(
        "Scaffold split sizes: "
        f"train={len(train_dataset)}, valid={len(valid_dataset)}, "
        f"test={len(test_dataset)}"
    )
    return train_dataset, valid_dataset, test_dataset


def to_tensor_dataset(dataset: Any) -> TensorDataset:
    features = torch.as_tensor(np.asarray(dataset.X), dtype=torch.float32)
    targets = torch.as_tensor(
        np.asarray(dataset.y).reshape(-1, 1), dtype=torch.float32
    )
    return TensorDataset(features, targets)


def make_loader(
    dataset: Any,
    *,
    shuffle: bool,
    seed: int = SEED,
    batch_size: int = BATCH_SIZE,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        to_tensor_dataset(dataset),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=0,
    )


def run_overfit_check(train_dataset: Any) -> dict[str, float | int]:
    """Overfit 32 samples to catch broken tensor/training-loop logic early."""
    subset_size = min(32, len(train_dataset))
    features = torch.as_tensor(
        np.asarray(train_dataset.X[:subset_size]), dtype=torch.float32
    )
    targets = torch.as_tensor(
        np.asarray(train_dataset.y[:subset_size]).reshape(-1, 1),
        dtype=torch.float32,
    )

    model = ESOLMLP(dropout=0.0).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    model.train()
    with torch.no_grad():
        initial_mse = float(criterion(model(features), targets).item())

    final_mse = initial_mse
    steps_run = 0
    for step in range(1, 1001):
        optimizer.zero_grad(set_to_none=True)
        predictions = model(features)
        loss = criterion(predictions, targets)
        loss.backward()
        optimizer.step()
        final_mse = float(loss.item())
        steps_run = step
        if final_mse <= 0.0025:
            break

    reduction_ratio = final_mse / max(initial_mse, np.finfo(float).eps)
    passed = final_mse <= 0.01 and reduction_ratio <= 0.01
    print(
        "32-sample overfit check: "
        f"initial MSE={initial_mse:.6f}, final MSE={final_mse:.6f}, "
        f"steps={steps_run}, passed={passed}"
    )
    if not passed:
        raise RuntimeError(
            "The 32-sample overfit check failed. Stop before formal training and "
            "inspect tensor shape/dtype, model.train(), zero_grad(), and backward()."
        )
    return {
        "samples": subset_size,
        "steps": steps_run,
        "initial_mse": initial_mse,
        "final_mse": final_mse,
        "reduction_ratio": reduction_ratio,
    }


@torch.no_grad()
def evaluate_loader(
    model: nn.Module, loader: DataLoader
) -> tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    prediction_batches: list[np.ndarray] = []
    target_batches: list[np.ndarray] = []
    squared_error_sum = 0.0
    sample_count = 0

    for features, targets in loader:
        predictions = model(features.to(DEVICE))
        targets = targets.to(DEVICE)
        squared_error_sum += float(
            torch.sum((predictions - targets) ** 2).item()
        )
        sample_count += int(targets.numel())
        prediction_batches.append(predictions.cpu().numpy().reshape(-1))
        target_batches.append(targets.cpu().numpy().reshape(-1))

    mse = squared_error_sum / sample_count
    return (
        float(np.sqrt(mse)),
        np.concatenate(target_batches),
        np.concatenate(prediction_batches),
    )


def train_with_early_stopping(
    train_dataset: Any, valid_dataset: Any
) -> tuple[ESOLMLP, dict[str, list[float]], int, float]:
    train_loader = make_loader(train_dataset, shuffle=True)
    train_eval_loader = make_loader(train_dataset, shuffle=False)
    valid_loader = make_loader(valid_dataset, shuffle=False)

    model = ESOLMLP(
        input_dim=INPUT_DIM,
        hidden_dim_1=HIDDEN_DIM_1,
        hidden_dim_2=HIDDEN_DIM_2,
        dropout=DROPOUT,
    ).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    history: dict[str, list[float]] = {"train_rmse": [], "valid_rmse": []}
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    best_valid_rmse = float("inf")
    epochs_without_improvement = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        for features, targets in train_loader:
            features = features.to(DEVICE)
            targets = targets.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            predictions = model(features)
            loss = criterion(predictions, targets)
            loss.backward()
            optimizer.step()

        train_rmse, _, _ = evaluate_loader(model, train_eval_loader)
        valid_rmse, _, _ = evaluate_loader(model, valid_loader)
        history["train_rmse"].append(train_rmse)
        history["valid_rmse"].append(valid_rmse)

        improved = valid_rmse < best_valid_rmse - 1e-8
        if improved:
            best_valid_rmse = valid_rmse
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        print(
            f"Epoch {epoch:03d} | train RMSE={train_rmse:.4f} | "
            f"valid RMSE={valid_rmse:.4f}"
            + (" | best" if improved else "")
        )

        if epochs_without_improvement >= PATIENCE:
            print(
                f"Early stopping at epoch {epoch}; "
                f"best epoch={best_epoch}, valid RMSE={best_valid_rmse:.4f}."
            )
            break

    if best_state is None:
        raise RuntimeError("Training produced no valid checkpoint.")
    model.load_state_dict(best_state)
    return model, history, best_epoch, best_valid_rmse


def regression_metrics(
    targets: np.ndarray, predictions: np.ndarray
) -> dict[str, float]:
    return {
        "rmse": float(np.sqrt(mean_squared_error(targets, predictions))),
        "mae": float(mean_absolute_error(targets, predictions)),
        "r2": float(r2_score(targets, predictions)),
    }


def save_checkpoint(
    model: ESOLMLP,
    *,
    best_epoch: int,
    best_valid_rmse: float,
    train_mean: float,
) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_path = OUTPUT_DIR / "best_model.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": {
                "input_dim": INPUT_DIM,
                "hidden_dim_1": HIDDEN_DIM_1,
                "hidden_dim_2": HIDDEN_DIM_2,
                "dropout": DROPOUT,
            },
            "featurizer": {
                "name": "ECFP",
                "size": INPUT_DIM,
                "radius": 2,
                "feature_standardization": False,
            },
            "training_config": {
                "seed": SEED,
                "learning_rate": LEARNING_RATE,
                "batch_size": BATCH_SIZE,
                "max_epochs": MAX_EPOCHS,
                "patience": PATIENCE,
                "loss": "MSELoss",
                "optimizer": "Adam",
                "device": "cpu",
            },
            "best_epoch": best_epoch,
            "best_valid_rmse": best_valid_rmse,
            "train_target_mean": train_mean,
        },
        checkpoint_path,
    )
    return checkpoint_path


def save_figures(
    history: dict[str, list[float]],
    targets: np.ndarray,
    predictions: np.ndarray,
    test_metrics: dict[str, float],
) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    epochs = np.arange(1, len(history["train_rmse"]) + 1)

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.plot(epochs, history["train_rmse"], label="Train RMSE", color="#0072B2")
    ax.plot(epochs, history["valid_rmse"], label="Validation RMSE", color="#D55E00")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("RMSE")
    ax.set_title("ESOL training history")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "loss_curve.png", dpi=200)
    plt.close(fig)

    lower = float(min(targets.min(), predictions.min()))
    upper = float(max(targets.max(), predictions.max()))
    padding = max((upper - lower) * 0.05, 0.1)
    limits = [lower - padding, upper + padding]

    fig, ax = plt.subplots(figsize=(5.8, 5.4))
    ax.scatter(
        targets,
        predictions,
        s=32,
        alpha=0.75,
        color="#009E73",
        edgecolors="white",
        linewidths=0.4,
    )
    ax.plot(limits, limits, linestyle="--", color="#333333", label="y = x")
    ax.set_xlim(limits)
    ax.set_ylim(limits)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Experimental logS")
    ax.set_ylabel("Predicted logS")
    ax.set_title(
        "Test-set predictions\n"
        f"RMSE={test_metrics['rmse']:.3f}, MAE={test_metrics['mae']:.3f}, "
        f"R²={test_metrics['r2']:.3f}"
    )
    ax.grid(alpha=0.20)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "test_predictions.png", dpi=200)
    plt.close(fig)


def main() -> None:
    set_reproducible_seed(SEED)
    train_dataset, valid_dataset, test_dataset = load_esol()
    overfit_check = run_overfit_check(train_dataset)

    train_mean = float(np.asarray(train_dataset.y).mean())
    model, history, best_epoch, best_valid_rmse = train_with_early_stopping(
        train_dataset, valid_dataset
    )
    checkpoint_path = save_checkpoint(
        model,
        best_epoch=best_epoch,
        best_valid_rmse=best_valid_rmse,
        train_mean=train_mean,
    )

    # The test set is touched only here, after model selection is complete.
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=True)
    final_model = ESOLMLP(**checkpoint["model_config"]).to(DEVICE)
    final_model.load_state_dict(checkpoint["model_state_dict"])
    test_loader = make_loader(test_dataset, shuffle=False)
    _, test_targets, test_predictions = evaluate_loader(final_model, test_loader)

    baseline_predictions = np.full_like(test_targets, train_mean, dtype=float)
    test_metrics = regression_metrics(test_targets, test_predictions)
    baseline_metrics = regression_metrics(test_targets, baseline_predictions)

    output = pd.DataFrame(
        {
            "smiles": np.asarray(test_dataset.ids).astype(str),
            "true_logS": test_targets,
            "predicted_logS": test_predictions,
        }
    )
    output.to_csv(OUTPUT_DIR / "test_predictions.csv", index=False)

    metrics = {
        "test": test_metrics,
        "train_mean_baseline": {
            "prediction": train_mean,
            **baseline_metrics,
        },
        "validation": {
            "best_epoch": best_epoch,
            "best_rmse": best_valid_rmse,
        },
        "data": {
            "dataset": "Delaney ESOL",
            "total_samples": 1128,
            "train_samples": len(train_dataset),
            "validation_samples": len(valid_dataset),
            "test_samples": len(test_dataset),
            "splitter": "scaffold",
            "feature": "1024-bit ECFP",
            "feature_standardization": False,
        },
        "overfit_check": overfit_check,
        "environment": {
            "python": ".".join(map(str, os.sys.version_info[:3])),
            "deepchem": dc.__version__,
            "torch": torch.__version__,
            "rdkit": rdkit.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
        },
    }
    with (OUTPUT_DIR / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, ensure_ascii=False)

    save_figures(history, test_targets, test_predictions, test_metrics)
    print(json.dumps(metrics["test"], indent=2))
    print(f"Artifacts written to: {OUTPUT_DIR}")
    print(f"Figures written to: {FIGURE_DIR}")


if __name__ == "__main__":
    main()
