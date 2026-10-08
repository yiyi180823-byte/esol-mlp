"""Predict ESOL logS for one valid SMILES string."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
os.environ.setdefault("DEEPCHEM_DATA_DIR", str(PROJECT_DIR / "data"))

import numpy as np
import torch
from rdkit import Chem, rdBase
from rdkit.Chem import rdFingerprintGenerator

from model import ESOLMLP


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict aqueous solubility (ESOL logS) from one SMILES string."
    )
    parser.add_argument("smiles", help="A valid SMILES string, for example CCO")
    parser.add_argument(
        "--model",
        type=Path,
        default=PROJECT_DIR / "outputs" / "best_model.pt",
        help="Path to the trained checkpoint.",
    )
    return parser.parse_args()


def predict_log_s(smiles: str, model_path: Path) -> float:
    with rdBase.BlockLogs():
        molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    if not model_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {model_path}. Run python train.py first."
        )

    checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    model = ESOLMLP(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    feature_config = checkpoint["featurizer"]
    size = int(feature_config["size"])
    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=int(feature_config["radius"]),
        includeChirality=False,
        useBondTypes=True,
        fpSize=size,
    )
    fingerprint = generator.GetFingerprint(molecule)
    features = np.asarray(fingerprint, dtype=np.float32).reshape(1, -1)
    if features.shape != (1, size):
        raise RuntimeError(f"Unexpected feature shape: {features.shape}")

    with torch.no_grad():
        prediction = model(torch.from_numpy(features)).item()
    return float(prediction)


def main() -> None:
    args = parse_args()
    try:
        prediction = predict_log_s(args.smiles, args.model)
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        raise SystemExit(str(exc)) from exc
    print(f"SMILES: {args.smiles}")
    print(f"Predicted logS: {prediction:.6f}")


if __name__ == "__main__":
    main()
