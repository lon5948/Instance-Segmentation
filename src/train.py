import math

import torch


def collate_fn(batch):
    return tuple(zip(*batch))


def train_one_epoch(
    model, loader, optimizer, device, epoch, print_freq=50, clip_value=1.0
):
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
                        for loss_name, loss in d.items():
                            if torch.is_tensor(loss) and loss.numel() == 1:
                                if torch.isfinite(loss).item():
                                    total_loss += loss
                                    valid_loss = True
            else:
                for loss_name, loss in loss_dict.items():
                    if torch.is_tensor(loss) and loss.numel() == 1:
                        if torch.isfinite(loss).item():
                            total_loss += loss
                            valid_loss = True

            # Only backprop if we have valid losses
            if valid_loss and torch.isfinite(total_loss).item():

                total_loss.backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)

                optimizer.step()

                loss_value = total_loss.item()
                running_loss += loss_value
                valid_steps += 1
            else:
                print(f"Skipping step {step} due to invalid loss value")
                loss_value = float("nan")

            # Print progress
            if step % print_freq == 0:
                if valid_steps > 0:
                    avg_loss = running_loss / valid_steps
                else:
                    avg_loss = float("nan")
                print(f"{header} step {step:04d} loss {avg_loss:.3f}")

                # If we've been encountering NaNs too frequently, reduce learning rate
                if math.isnan(avg_loss) and valid_steps < step / 2:
                    for param_group in optimizer.param_groups:
                        param_group["lr"] *= 0.5
                    print(
                        f"Reducing learning rate to {optimizer.param_groups[0]['lr']}"
                    )

        except Exception as e:
            print(f"Error in training step {step}: {e}")
            continue


@torch.no_grad()
def evaluate(model, loader, device):
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
