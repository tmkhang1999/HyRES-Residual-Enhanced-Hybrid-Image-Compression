import time
import torch
from src.losses import AverageMeter


def train_one_epoch(
        model, criterion, train_dataloader, optimizer, aux_optimizer, epoch, clip_max_norm, noisequant=True,
):
    model.train()
    device = next(model.parameters()).device
    train_loss = AverageMeter()
    train_bpp_loss = AverageMeter()
    train_y_bpp_loss = AverageMeter()
    train_z_bpp_loss = AverageMeter()
    train_mse_loss = AverageMeter()
    start = time.time()

    for i, d in enumerate(train_dataloader):
        d = d.to(device)

        optimizer.zero_grad()
        aux_optimizer.zero_grad()

        # Process with model
        out_net = model(d, noisequant)

        out_criterion = criterion(out_net, d)
        train_bpp_loss.update(out_criterion["bpp_loss"].item())
        train_y_bpp_loss.update(out_criterion["y_bpp_loss"].item())
        train_z_bpp_loss.update(out_criterion["z_bpp_loss"].item())
        train_loss.update(out_criterion["loss"].item())
        train_mse_loss.update(out_criterion["mse_loss"].item())

        out_criterion["loss"].backward()
        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
        optimizer.step()

        aux_loss = model.aux_loss()
        aux_loss.backward()
        aux_optimizer.step()

        if i % 100 == 0:
            print(
                f"Train epoch {epoch}: ["
                f"{i * len(d)}/{len(train_dataloader.dataset)}"
                f" ({100. * i / len(train_dataloader):.0f}%)]"
                f'\tLoss: {out_criterion["loss"].item():.3f} |'
                f'\tMSE loss: {out_criterion["mse_loss"].item():.3f} |'
                f'\tBpp loss: {out_criterion["bpp_loss"].item():.3f} |'
                f'\ty_Bpp loss: {out_criterion["y_bpp_loss"].item():.4f} |'
                f'\tz_Bpp loss: {out_criterion["z_bpp_loss"].item():.4f} |'
                f"\tAux loss: {aux_loss.item():.2f}"
            )
    print(f"Train epoch {epoch}: Average losses:"
          f"\tLoss: {train_loss.avg:.3f} |"
          f"\tMSE loss: {train_mse_loss.avg:.3f} |"
          f"\tBpp loss: {train_bpp_loss.avg:.4f} |"
          f"\ty_Bpp loss: {train_y_bpp_loss.avg:.5f} |"
          f"\tz_Bpp loss: {train_z_bpp_loss.avg:.5f} |"
          f"\tTime (s) : {time.time() - start:.4f} |"
          )

    return train_loss.avg, train_bpp_loss.avg, train_mse_loss.avg


def test_epoch(epoch, test_dataloader, model, criterion):
    model.eval()
    device = next(model.parameters()).device

    loss = AverageMeter()
    bpp_loss = AverageMeter()
    y_bpp_loss = AverageMeter()
    z_bpp_loss = AverageMeter()
    mse_loss = AverageMeter()
    aux_loss = AverageMeter()

    with torch.no_grad():
        for d in test_dataloader:
            d = d.to(device)
            out_net = model(d)
            out_criterion = criterion(out_net, d)

            aux_loss.update(model.aux_loss().item())
            bpp_loss.update(out_criterion["bpp_loss"].item())
            y_bpp_loss.update(out_criterion["y_bpp_loss"].item())
            z_bpp_loss.update(out_criterion["z_bpp_loss"].item())
            loss.update(out_criterion["loss"].item())
            mse_loss.update(out_criterion["mse_loss"].item())

    print(
        f"Test epoch {epoch}: Average losses:"
        f"\tLoss: {loss.avg:.3f} |"
        f"\tMSE loss: {mse_loss.avg:.3f} |"
        f"\tBpp loss: {bpp_loss.avg:.4f} |"
        f"\ty_Bpp loss: {y_bpp_loss.avg:.4f} |"
        f"\tz_Bpp loss: {z_bpp_loss.avg:.4f} |"
        f"\tAux loss: {aux_loss.avg:.4f}\n"
    )

    return loss.avg, bpp_loss.avg, mse_loss.avg
