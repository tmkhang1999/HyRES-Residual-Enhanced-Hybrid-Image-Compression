import os
import time

import torch
import math
from src.utils import AverageMeter
from torchvision.utils import save_image


def train_one_epoch(model, criterion, train_dataloader, optimizer, epoch):
    model.train()
    device = next(model.parameters()).device
    train_loss = AverageMeter()
    train_mse = AverageMeter()
    train_jpeg_bpp = AverageMeter()
    train_residual_bpp = AverageMeter()
    start = time.time()

    for i, data in enumerate(train_dataloader):
        # Handle the case where data might be a tuple or just images
        if isinstance(data, tuple) or isinstance(data, list):
            images = data[0]
        else:
            images = data

        images = images.to(device)

        # Forward pass
        results = model(images, noisequant=False)
        x_hat = results['x_hat']

        # Calculate MSE loss (convert to 255 scale)
        loss = criterion(x_hat, images)
        loss = loss * 255 ** 2

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update metrics
        train_loss.update(loss.item())
        train_mse.update(loss.item())  # Same as loss for MSE

        # Track bpp values if available
        if 'jpeg_bpp_loss' in results:
            train_jpeg_bpp.update(results['jpeg_bpp_loss'].item())

        if 'likelihoods' in results:
            # Calculate residual bpp from likelihoods (simplified estimate)
            N, _, H, W = images.size()
            num_pixels = N * H * W
            residual_bpp = sum(
                (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
                for likelihoods in results["likelihoods"].values()
            )
            train_residual_bpp.update(residual_bpp.item())

        if (i + 1) % 100 == 0:
            print(f"Epoch [{epoch + 1}], Step [{i + 1}/{len(train_dataloader)}], "
                  f"Loss: {loss.item():.6f}, "
                  f"MSE: {loss.item():.6f}, "
                  f"JPEG BPP: {train_jpeg_bpp.avg:.4f}, "
                  f"Residual BPP: {train_residual_bpp.avg:.4f}")

    print(f"Train epoch {epoch + 1}: "
          f"Average MSE loss: {train_mse.avg:.6f} | "
          f"JPEG BPP: {train_jpeg_bpp.avg:.4f} | "
          f"Residual BPP: {train_residual_bpp.avg:.4f} | "
          f"Time (s): {time.time() - start:.4f}")

    return train_loss.avg


def validate(model, criterion, test_dataloader, epoch, save_images=False, savepath=None):
    model.eval()
    device = next(model.parameters()).device

    val_loss = AverageMeter()
    val_jpeg_bpp = AverageMeter()
    val_residual_bpp = AverageMeter()

    start = time.time()

    # Set up directory for reconstructed images
    recon_dir = os.path.join(savepath, "best_recon") if save_images else None
    if save_images and os.path.exists(recon_dir):
        import shutil
        shutil.rmtree(recon_dir)
    if save_images:
        os.makedirs(recon_dir, exist_ok=True)

    with torch.no_grad():
        for i, data in enumerate(test_dataloader):
            # Handle the case where data might be a tuple or just images
            if isinstance(data, tuple) or isinstance(data, list):
                images = data[0]
            else:
                images = data

            images = images.to(device)

            # Forward pass
            results = model(images, noisequant=False)
            x_hat = results['x_hat']

            # Calculate MSE loss (convert to 255 scale)
            loss = criterion(x_hat, images)
            loss = loss * 255 ** 2

            # Update metrics
            val_loss.update(loss.item())

            # Track bpp values if available
            if 'jpeg_bpp_loss' in results:
                val_jpeg_bpp.update(results['jpeg_bpp_loss'].item())

            if 'likelihoods' in results:
                # Calculate residual bpp from likelihoods
                N, _, H, W = images.size()
                num_pixels = N * H * W
                residual_bpp = sum(
                    (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
                    for likelihoods in results["likelihoods"].values()
                )
                val_residual_bpp.update(residual_bpp.item())

            # Save reconstructed images (only for the first 6)
            if save_images and i < 6:
                # Save original image
                original_path = os.path.join(recon_dir, f"original_{i}.png")
                save_image(images, original_path)

                # Save reconstructed image
                recon_path = os.path.join(recon_dir, f"recon_{i}.png")
                save_image(x_hat, recon_path)

                # Save JPEG reconstruction
                jpeg_path = os.path.join(recon_dir, f"jpeg_{i}.png")
                save_image(results["jpeg_decoded"], jpeg_path)

                # Save enhancement components if available
                if "residual" in results:
                    residual_path = os.path.join(recon_dir, f"residual_{i}.png")
                    residual_vis = results["residual"] * 0.5 + 0.5  # Normalize for visualization
                    save_image(residual_vis, residual_path)

                if "residual_hat" in results:
                    residual_hat_path = os.path.join(recon_dir, f"residual_hat_{i}.png")
                    residual_hat_vis = results["residual_hat"] * 0.5 + 0.5
                    save_image(residual_hat_vis, residual_hat_path)

    print(f"Validation epoch {epoch + 1}: "
          f"Average MSE loss: {val_loss.avg:.6f} | "
          f"JPEG BPP: {val_jpeg_bpp.avg:.4f} | "
          f"Residual BPP: {val_residual_bpp.avg:.4f} | "
          f"Time (s): {time.time() - start:.4f}")

    return val_loss.avg