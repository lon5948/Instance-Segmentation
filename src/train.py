import math
from pathlib import Path

import matplotlib.pyplot as plt
import torch


def collate_fn(batch):
    return tuple(zip(*batch))


def plot_losses(train_losses, val_losses, save_dir=".", filename="loss_curve.png"):
    """Plot and save training & validation loss curves.

    Args:
        train_losses (List[float]): list of average training losses per epoch.
        val_losses   (List[float]): list of average validation losses per epoch.
        save_dir (str | Path): directory to save the image.
        filename (str): name of the image file.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    epochs = range(1, len(train_losses) + 1)
    plt.figure()
    plt.plot(epochs, train_losses, label="Train")
    plt.plot(epochs, val_losses, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs. Validation Loss")
    plt.legend()
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(save_dir / filename)
    plt.close()
    print(f"[Plot] Saved loss curve → {save_dir / filename}")


def train_one_epoch(
    model, loader, optimizer, device, epoch, print_freq=50, clip_value=1.0
):
    """
    Trains the model for a single epoch and returns the average loss.

    Returns:
        float: Average training loss for the epoch.
    """
    model.train()
    header = f"Epoch [{epoch}]"
    running_loss = 0.0
    valid_steps = 0

    for step, (images, targets) in enumerate(loader):
        try:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # Zero gradients before forward pass
            optimizer.zero_grad()

            # Forward pass
            loss_dict = model(images, targets)

            # Process the loss dictionary safely
            total_loss = 0.0
            valid_loss = False

            if isinstance(loss_dict, list):
                for d in loss_dict:
                    if hasattr(d, "values"):
                        for loss in d.values():
                            if torch.is_tensor(loss) and loss.numel() == 1:
                                if torch.isfinite(loss).item():
                                    total_loss += loss
                                    valid_loss = True
            else:
                for loss in loss_dict.values():
                    if torch.is_tensor(loss) and loss.numel() == 1:
                        if torch.isfinite(loss).item():
                            total_loss += loss
                            valid_loss = True

            if valid_loss and torch.isfinite(total_loss).item():
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
                optimizer.step()

                running_loss += total_loss.item()
                valid_steps += 1
            else:
                print(f"Skipping step {step} due to invalid loss value")

            if step % print_freq == 0:
                avg_loss = (
                    running_loss / valid_steps if valid_steps > 0 else float("nan")
                )
                print(f"{header} step {step:04d} loss {avg_loss:.3f}")

                if math.isnan(avg_loss) and valid_steps < step / 2:
                    for param_group in optimizer.param_groups:
                        param_group["lr"] *= 0.5
                    print(
                        f"Reducing learning rate to {optimizer.param_groups[0]['lr']}"
                    )

        except Exception as e:
            print(f"Error in training step {step}: {e}")
            continue

    epoch_loss = running_loss / valid_steps if valid_steps > 0 else float("inf")
    print(f"{header} average loss {epoch_loss:.4f}")
    return epoch_loss


@torch.no_grad()
def evaluate(model, loader, device):
    """
    Evaluates the model on the validation set and returns average loss.
    """
    running = 0.0
    num_valid_batches = 0

    for images, targets in loader:
        try:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)

            batch_loss = 0
            has_valid_loss = False

            if isinstance(loss_dict, list):
                for d in loss_dict:
                    if hasattr(d, "values"):
                        for loss in d.values():
                            if torch.is_tensor(loss) and loss.numel() == 1:
                                if torch.isfinite(loss).item():
                                    batch_loss += loss.item()
                                    has_valid_loss = True
            else:
                for loss in loss_dict.values():
                    if torch.is_tensor(loss) and loss.numel() == 1:
                        if torch.isfinite(loss).item():
                            batch_loss += loss.item()
                            has_valid_loss = True

            if has_valid_loss:
                running += batch_loss
                num_valid_batches += 1

        except Exception as e:
            print(f"Skipping batch during evaluation due to error: {e}")
            continue

    if num_valid_batches > 0:
        avg_val_loss = running / num_valid_batches
        print(f"[Val] Avg Loss: {avg_val_loss:.4f} over {num_valid_batches} batches")
        return avg_val_loss
    else:
        print("[Val] No valid batches, returning loss=inf")
        return float("inf")
