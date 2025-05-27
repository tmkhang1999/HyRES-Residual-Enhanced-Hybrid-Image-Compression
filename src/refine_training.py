import argparse
import os
import random

import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torchvision import transforms

from models.checkerboard import LightWeightCheckerboard
from models.hyres import ResidualJPEGCompression
from src.utils import ImageFolder, save_checkpoint
from src.utils.refine_engine import train_one_epoch, validate


class PostProcessTrainer:
    def __init__(self, args):
        self.args = args
        self.device = self._setup_device()
        self.model = self._setup_model()
        self.train_loader, self.test_loader = self._setup_dataloaders()
        self.optimizer = self._setup_optimizer()
        self.criterion = nn.MSELoss()
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.1, patience=10, verbose=True
        )
        self.writer = SummaryWriter(args.savepath)

    def _setup_device(self):
        if self.args.cuda:
            if torch.cuda.is_available():
                device = torch.device("cuda")
                print("Using CUDA")
                return device
            elif torch.backends.mps.is_available():
                device = torch.device("mps")
                print("Using MPS")
                return device
        device = torch.device("cpu")
        print("Using CPU")
        return device

    def _setup_model(self):
        # Create base model
        base_model = LightWeightCheckerboard(N=self.args.N, M=self.args.M)

        # Create full model with SE and refinement blocks
        model = ResidualJPEGCompression(
            base_model=base_model,
            jpeg_quality=self.args.jpeg_quality,
            se_reduction=self.args.se_reduction
        )

        # Load pretrained weights if specified
        if self.args.checkpoint:
            print(f"Loading pretrained model from {self.args.checkpoint}")
            checkpoint = torch.load(self.args.checkpoint, map_location=self.device)

            # If the checkpoint doesn't have se_block and refine, it's a base model only
            if any(k.startswith('se_block.') for k in checkpoint["state_dict"]):
                # Full model checkpoint
                model.load_state_dict(checkpoint["state_dict"])
            else:
                # Base model only checkpoint
                model.load_state_dict(checkpoint["state_dict"], strict=False)

        # Freeze base model parameters
        for name, param in model.named_parameters():
            if not (name.startswith('se_block.') or name.startswith('refine.')):
                param.requires_grad = False

        print(f"Model initialized. Trainable parameters:")
        trainable_params = 0
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(f" - {name}: {param.size()}")
                trainable_params += param.numel()
        print(f"Total trainable parameters: {trainable_params}")

        return model.to(self.device)

    def _setup_dataloaders(self):
        # Set up data transforms
        train_transforms = transforms.Compose([
            transforms.RandomCrop(self.args.patch_size),
            transforms.ToTensor()
        ])

        test_transforms = transforms.Compose([
            transforms.ToTensor()
        ])

        # Create datasets
        train_dataset = ImageFolder(
            self.args.dataset,
            split="train",
            transform=train_transforms
        )

        test_dataset = ImageFolder(
            self.args.dataset,
            split="test",
            transform=test_transforms
        )

        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            shuffle=True,
            pin_memory=(self.device.type == "cuda")
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=self.args.test_batch_size,
            num_workers=self.args.num_workers,
            shuffle=False,
            pin_memory=(self.device.type == "cuda")
        )

        return train_loader, test_loader

    def _setup_optimizer(self):
        # Only optimize SE block and refinement block parameters
        params_to_optimize = []
        for name, param in self.model.named_parameters():
            if param.requires_grad:  # Only add trainable parameters
                params_to_optimize.append(param)

        return optim.Adam(params_to_optimize, lr=self.args.learning_rate)

    def save_model(self, epoch, loss, is_best=False):
        # Save only the SE block and refinement parameters
        se_block_state_dict = {
            k: v for k, v in self.model.state_dict().items()
            if k.startswith('se_block.') or k.startswith('refine.')
        }

        checkpoint = {
            'epoch': epoch,
            'state_dict': se_block_state_dict,
            'loss': loss,
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
        }

        # Save last checkpoint with epoch number
        filename = os.path.join(
            self.args.savepath,
            f"checkpoint_last_{epoch}.pth.tar"
        )
        save_checkpoint(checkpoint, filename=filename)

        # Remove previous last checkpoints
        for file in os.listdir(self.args.savepath):
            if file.startswith("checkpoint_last_") and file != f"checkpoint_last_{epoch}.pth.tar":
                os.remove(os.path.join(self.args.savepath, file))

        # Save best checkpoint if needed with epoch number
        if is_best:
            best_filename = os.path.join(
                self.args.savepath,
                f"checkpoint_best_{epoch}.pth.tar"
            )
            save_checkpoint(checkpoint, filename=best_filename)

            # Remove previous best checkpoints
            for file in os.listdir(self.args.savepath):
                if file.startswith("checkpoint_best_") and file != f"checkpoint_best_{epoch}.pth.tar":
                    os.remove(os.path.join(self.args.savepath, file))

    def train(self):
        best_loss = float('inf')
        start_epoch = 0

        # Load checkpoint if continuing training
        if self.args.resume:
            if os.path.isfile(self.args.resume):
                print(f"Loading checkpoint '{self.args.resume}'")
                checkpoint = torch.load(self.args.resume, map_location=self.device)

                # Load only SE block and refinement parameters
                se_block_state_dict = {}
                for k, v in checkpoint["state_dict"].items():
                    if k.startswith('se_block.') or k.startswith('refine.'):
                        se_block_state_dict[k] = v

                # Partial load of state dict
                for name, param in self.model.named_parameters():
                    if name in se_block_state_dict:
                        param.data = se_block_state_dict[name]

                start_epoch = checkpoint["epoch"] + 1
                self.optimizer.load_state_dict(checkpoint["optimizer"])
                self.scheduler.load_state_dict(checkpoint["scheduler"])
                best_loss = checkpoint["loss"]
                print(f"Loaded checkpoint '{self.args.resume}' (epoch {checkpoint['epoch']})")
            else:
                print(f"No checkpoint found at '{self.args.resume}'")

        for epoch in range(start_epoch, self.args.epochs):
            print(f"Epoch {epoch + 1}/{self.args.epochs}")
            print(f"Learning rate: {self.optimizer.param_groups[0]['lr']}")

            # Train one epoch
            train_loss = train_one_epoch(
                self.model,
                self.criterion,
                self.train_loader,
                self.optimizer,
                epoch
            )
            self.writer.add_scalar('Train/MSE', train_loss, epoch)

            # Validate
            val_loss = validate(
                self.model,
                self.criterion,
                self.test_loader,
                epoch)
            self.writer.add_scalar('Validation/MSE', val_loss, epoch)

            # Update learning rate scheduler
            self.scheduler.step(val_loss)

            # Save model
            is_best = val_loss < best_loss
            if is_best:
                best_loss = val_loss
                print(f"New best model found! MSE: {best_loss:.6f}")
                # Save reconstructions for best model
                val_loss = validate(
                    self.model,
                    self.criterion,
                    self.test_loader,
                    epoch,
                    save_images=True,
                    savepath=self.args.savepath)

            self.save_model(epoch, val_loss, is_best)
            print("-" * 80)


def parse_args():
    parser = argparse.ArgumentParser(description="Train post-processing blocks (SE and refinement).")
    parser.add_argument("--dataset", type=str, required=True, help="Training dataset")
    parser.add_argument("--N", default=128, type=int, help="Number of channels")
    parser.add_argument("--M", default=192, type=int, help="Number of latent channels")
    parser.add_argument("--jpeg-quality", default=1, type=int, help="JPEG quality factor")
    parser.add_argument("--se-reduction", default=16, type=int, help="SE block reduction factor")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--test-batch-size", type=int, default=32, help="Test batch size")
    parser.add_argument("--patch-size", type=int, nargs=2, default=(256, 256), help="Training patch size")
    parser.add_argument("--num-workers", type=int, default=4, help="Dataloader workers")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--checkpoint", type=str, help="Path to pretrained model checkpoint")
    parser.add_argument("--resume", type=str, help="Path to resume training from a checkpoint")
    parser.add_argument("--savepath", default="./checkpoint/enhancement", type=str, help="Path to save checkpoints")
    parser.add_argument("--cuda", type=lambda x: str(x).lower() == 'true', default=True, help="Use CUDA")
    parser.add_argument("--seed", default=1926, type=int, help="Random seed")

    return parser.parse_args()


def main():
    args = parse_args()

    # Set random seed
    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = False

    # Create directories
    os.makedirs(args.savepath, exist_ok=True)

    # Initialize trainer and start training
    trainer = PostProcessTrainer(args)
    trainer.train()


if __name__ == "__main__":
    main()