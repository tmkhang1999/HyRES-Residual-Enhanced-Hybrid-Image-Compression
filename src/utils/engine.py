import time
import torch
import os
from src.losses import AverageMeter
from contextlib import nullcontext


def train_one_epoch(
        model, criterion, train_dataloader, optimizer, aux_optimizer, epoch, clip_max_norm,
        noisequant=True, mixed_precision=False, gradient_accumulation_steps=1,
):
    model.train()
    device = next(model.parameters()).device
    train_loss = AverageMeter()
    train_bpp_loss = AverageMeter()
    train_y_bpp_loss = AverageMeter()
    train_z_bpp_loss = AverageMeter()
    train_mse_loss = AverageMeter()
    train_vgg_loss = AverageMeter()
    start = time.time()

    # Initialize scaler for mixed precision
    scaler = torch.cuda.amp.GradScaler() if mixed_precision else None

    optimizer.zero_grad()
    aux_optimizer.zero_grad()
    aux_loss = torch.tensor(0.0, device=device)

    for i, d in enumerate(train_dataloader):
        # Keep data on CPU for JPEG compression, then move to GPU for neural compression
        # Process with mixed precision
        with torch.cuda.amp.autocast() if mixed_precision else nullcontext():
            out_net = model(d, noisequant)

            # Move data to GPU after JPEG compression
            d = d.to(device)

            out_criterion = criterion(out_net, d)
            loss = out_criterion["loss"] / gradient_accumulation_steps

        # Accumulate metrics
        train_bpp_loss.update(out_criterion["bpp_loss"].item())
        train_y_bpp_loss.update(out_criterion["y_bpp_loss"].item())
        train_z_bpp_loss.update(out_criterion["z_bpp_loss"].item())
        train_loss.update(out_criterion["loss"].item())
        train_mse_loss.update(out_criterion["mse_loss"].item())
        train_vgg_loss.update(out_criterion["vgg_loss"].item())

        # Scale loss and backward pass with mixed precision
        if mixed_precision:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        # Perform optimization step after accumulating gradients
        if (i + 1) % gradient_accumulation_steps == 0:
            if clip_max_norm > 0:
                if mixed_precision:
                    scaler.unscale_(optimizer)
                    # Check for NaN gradients and skip step if found
                    skip_step = False
                    for param in model.parameters():
                        if param.grad is not None and torch.isnan(param.grad).any():
                            skip_step = True
                            break

                    if not skip_step:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
                    else:
                        print(f"Warning: NaN gradients detected, skipping update step")
                        optimizer.zero_grad()
                        if scaler is not None:
                            scaler.update()
                        continue
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)

            if mixed_precision:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            optimizer.zero_grad()

            # Handle auxiliary optimizer separately without mixed precision
            aux_loss = model.aux_loss()
            aux_loss.backward()
            aux_optimizer.step()
            aux_optimizer.zero_grad()

        if i % 100 == 0:
            print(
                f"Train epoch {epoch}: ["
                f"{i * len(d)}/{len(train_dataloader.dataset)}"
                f" ({100. * i / len(train_dataloader):.0f}%)]"
                f'\tLoss: {out_criterion["loss"].item():.3f} |'
                f'\tMSE loss: {out_criterion["mse_loss"].item():.3f} |'
                f'\tVGG loss: {out_criterion["vgg_loss"].item():.3f} |'
                f'\tBpp loss: {out_criterion["bpp_loss"].item():.3f} |'
                f'\tResidual Bpp: {out_criterion["residual_bpp_loss"].item():.3f} |'
                f'\ty_Bpp loss: {out_criterion["y_bpp_loss"].item():.4f} |'
                f'\tz_Bpp loss: {out_criterion["z_bpp_loss"].item():.4f} |'
                f"\tAux loss: {aux_loss.item():.2f}"
            )
    print(f"Train epoch {epoch}: Average losses:"
          f"\tLoss: {train_loss.avg:.3f} |"
          f"\tMSE loss: {train_mse_loss.avg:.3f} |"
          f"\tVGG loss: {train_vgg_loss.avg:.3f} |"
          f"\tBpp loss: {train_bpp_loss.avg:.4f} |"
          f"\ty_Bpp loss: {train_y_bpp_loss.avg:.5f} |"
          f"\tz_Bpp loss: {train_z_bpp_loss.avg:.5f} |"
          f"\tTime (s) : {time.time() - start:.4f} |"
          )

    return train_loss.avg, train_bpp_loss.avg, train_mse_loss.avg


def test_epoch(epoch, test_dataloader, model, criterion, save_images=False, savepath=None):
    model.eval()
    device = next(model.parameters()).device

    loss = AverageMeter()
    bpp_loss = AverageMeter()
    y_bpp_loss = AverageMeter()
    z_bpp_loss = AverageMeter()
    mse_loss = AverageMeter()
    vgg_loss = AverageMeter()
    aux_loss = AverageMeter()

    # Set up directory for reconstructed images
    recon_dir = os.path.join(savepath, "best_recon") if save_images else None
    if save_images and os.path.exists(recon_dir):
        # Remove existing images to save only the latest best ones
        import shutil
        shutil.rmtree(recon_dir)
    if save_images:
        os.makedirs(recon_dir)

    with torch.no_grad():
        for i, d in enumerate(test_dataloader):
            # Process with the HyRes model's expected data flow
            out_net = model(d)

            # Move data to device for criterion calculation
            d = d.to(device)
            out_criterion = criterion(out_net, d)

            aux_loss.update(model.aux_loss().item())
            bpp_loss.update(out_criterion["bpp_loss"].item())
            y_bpp_loss.update(out_criterion["y_bpp_loss"].item())
            z_bpp_loss.update(out_criterion["z_bpp_loss"].item())
            loss.update(out_criterion["loss"].item())
            mse_loss.update(out_criterion["mse_loss"].item())
            vgg_loss.update(out_criterion["vgg_loss"].item())

            # Save reconstructed images if requested (only for the latest best epoch)
            if save_images and i < 5:  # Limit to first 20 images
                from torchvision.utils import save_image

                # Save reconstructed image
                img_path = os.path.join(recon_dir, f"recon_{i}.png")
                save_image(out_net["x_hat"], img_path)

                # For HyRes model, save component images too
                if "jpeg_decoded" in out_net and "residual_hat" in out_net:
                    jpeg_path = os.path.join(recon_dir, f"jpeg_{i}.png")
                    save_image(out_net["jpeg_decoded"], jpeg_path)

                    residual_path = os.path.join(recon_dir, f"residual_{i}.png")
                    # Normalize residual for better visualization (centered at 0.5)
                    residual_vis = out_net["residual"] * 0.5 + 0.5
                    save_image(residual_vis, residual_path)

                    residual_hat_path = os.path.join(recon_dir, f"residual_hat_{i}.png")
                    residual_hat_vis = out_net["residual_hat"] * 0.5 + 0.5
                    save_image(residual_hat_vis, residual_hat_path)

    print(
        f"Test epoch {epoch}: Average losses:"
        f"\tLoss: {loss.avg:.3f} |"
        f"\tMSE loss: {mse_loss.avg:.3f} |"
        f"\tVGG loss: {vgg_loss.avg:.3f} |"
        f"\tBpp loss: {bpp_loss.avg:.4f} |"
        f"\ty_Bpp loss: {y_bpp_loss.avg:.4f} |"
        f"\tz_Bpp loss: {z_bpp_loss.avg:.4f} |"
        f"\tAux loss: {aux_loss.avg:.4f}\n"
    )

    # Save metrics to CSV if requested (overwriting previous best)
    if save_images:
        import csv
        csv_path = os.path.join(savepath, "best_metrics.csv")
        with open(csv_path, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'loss', 'mse_loss', 'vgg_loss', 'bpp_loss', 'y_bpp_loss', 'z_bpp_loss', 'aux_loss'])
            writer.writerow([epoch, loss.avg, mse_loss.avg, vgg_loss.avg, bpp_loss.avg, y_bpp_loss.avg, z_bpp_loss.avg, aux_loss.avg])

    return loss.avg, bpp_loss.avg, mse_loss.avg
