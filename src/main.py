from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import DataLoader, random_split

from src.dataset import CellInstanceDataset
from src.model import get_instance_segmentation_model
from src.test import run_prediction
from src.train import collate_fn, evaluate, plot_losses, train_one_epoch
from src.transforms import get_test_transforms, get_train_transforms
from src.utils import set_seed


def parse_args():
    ap = argparse.ArgumentParser(description="VRDL HW3: Mask-RCNN Trainer")
    ap.add_argument(
        "--data_root",
        required=True,
        type=Path,
        help="Path to train/ or test/ root folder",
    )
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=0.0005)  # Reduced learning rate
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--output_dir", type=Path, default=Path("./checkpoints"))
    ap.add_argument(
        "--weights",
        type=Path,
        help="Path to a .pth checkpoint to resume / for inference",
    )
    ap.add_argument(
        "--predict_only",
        action="store_true",
        help="Skip training → run inference on given data_root",
    )
    ap.add_argument(
        "--outfile",
        type=str,
        default="test-results.json",
        help="Where to save prediction JSON",
    )
    ap.add_argument(
        "--id_map_json",
        type=Path,
        default="test_image_name_to_ids.json",
        help="Path to test_image_name_to_ids.json",
    )
    return ap.parse_args()


def main():
    args = parse_args()
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.predict_only:
        # ---------------- inference ----------------
        id_map: Dict[str, int] = {}
        map_path = Path(os.path.join(args.data_root, args.id_map_json))
        if map_path.exists():
            print(f"→ loading id_map from {map_path}")
            raw = map_path.read_text()
            obj = json.loads(raw)
            id_map = {d["file_name"]: d["id"] for d in obj}

        test_data_root = os.path.join(args.data_root, "test")
        test_ds = CellInstanceDataset(
            test_data_root, transforms=get_test_transforms(), train=False, id_map=id_map
        )
        test_loader = DataLoader(
            test_ds, batch_size=1, shuffle=False, collate_fn=collate_fn
        )

        model = get_instance_segmentation_model()
        assert args.weights is not None, "--weights required for prediction"
        model.load_state_dict(torch.load(args.weights, map_location=device))
        model.to(device)

        print(f"Starting inference with device: {device}")
        run_prediction(model, test_loader, device, args.outfile)
        return

    # ---------------- training --------------------
    train_data_root = os.path.join(args.data_root, "train")
    full_ds = CellInstanceDataset(
        train_data_root, transforms=get_train_transforms(), train=True
    )
    n_val = max(1, int(0.1 * len(full_ds)))
    n_train = len(full_ds) - n_val
    train_ds, val_ds = random_split(full_ds, [n_train, n_val])

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, collate_fn=collate_fn)

    model = get_instance_segmentation_model()

    trainable_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_param/1e6:.2f}M")

    if args.weights and args.weights.exists():
        print(f"→ resume from {args.weights}")
        model.load_state_dict(torch.load(args.weights, map_location=device))

    model.to(device)

    # Use Adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # Use ReduceLROnPlateau scheduler which reduces LR when validation loss plateaus
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    best_loss = float("inf")
    train_losses, val_losses = [], []

    print(f"Starting training with device: {device}")
    for epoch in range(1, args.epochs + 1):

        epoch_train_loss = train_one_epoch(
            model, train_loader, optimizer, device, epoch
        )
        train_losses.append(epoch_train_loss)

        try:
            val_loss = evaluate(model, val_loader, device)
            lr_scheduler.step(val_loss)
            val_losses.append(val_loss)

            if val_loss < best_loss:
                best_loss = val_loss
                best_ckpt = args.output_dir / "model_best.pth"
                torch.save(model.state_dict(), best_ckpt)
                print(f"✓ saved best model with loss {val_loss:.4f} → {best_ckpt}")

        except Exception as e:
            print(f"Evaluation error: {e}, continuing training")
            ckpt = args.output_dir / f"model_epoch_{epoch}.pth"
            torch.save(model.state_dict(), ckpt)
            print(f"✓ saved checkpoint → {ckpt}")

    plot_losses(train_losses, val_losses)


if __name__ == "__main__":
    main()
