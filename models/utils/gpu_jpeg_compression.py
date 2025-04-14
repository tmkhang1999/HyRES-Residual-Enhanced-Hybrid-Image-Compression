import torch

try:
    import cupy as cp
    import nvjpeg

    HAS_NVJPEG = True
except ImportError:
    HAS_NVJPEG = False
    print("Warning: nvJPEG not available. Install with: pip install nvidia-nvjpeg-cu11")


class NvJPEGCompression(torch.nn.Module):
    """JPEG compression using NVIDIA's nvJPEG library to run on GPU"""

    def __init__(self, quality=75):
        super().__init__()
        self.quality = quality

        # Check if nvJPEG is available
        if not HAS_NVJPEG:
            from ..utils.jpeg_compression import JPEGCompression
            self.cpu_fallback = JPEGCompression(quality=quality)
            print("Using CPU JPEG fallback")
        else:
            # Initialize nvJPEG encoder/decoder
            self.encoder = nvjpeg.Encoder(quality=quality)
            self.decoder = nvjpeg.Decoder()
            print(f"Using GPU JPEG compression with quality {quality}")

    def __call__(self, x):
        """Apply JPEG compression/decompression

        Args:
            x (torch.Tensor): Input batch of images [B,C,H,W] with values in [0,1]

        Returns:
            torch.Tensor: JPEG compressed/decompressed images
        """
        # If nvJPEG not available, fall back to CPU implementation
        if not HAS_NVJPEG:
            return self.cpu_fallback(x)

        # Get original device
        device = x.device
        batch_size, channels, height, width = x.shape

        # Prepare output tensor
        result = torch.empty_like(x)

        # Process batch
        for i in range(batch_size):
            # Convert to format expected by nvJPEG (HWC layout)
            img = x[i].permute(1, 2, 0).mul(255).byte()

            # Keep data on GPU with CuPy
            if device.type == 'cuda':
                img_cp = cp.asarray(img.cpu().numpy())

                # Encode to JPEG
                jpeg_bytes = self.encoder.encode(img_cp)

                # Decode JPEG
                decoded = self.decoder.decode(jpeg_bytes)

                # Convert back to torch tensor
                decoded_tensor = torch.from_numpy(cp.asnumpy(decoded)).float() / 255.0
            else:
                # Fall back to CPU for non-CUDA devices
                img_np = img.numpy()
                jpeg_bytes = self.encoder.encode(img_np)
                decoded = self.decoder.decode(jpeg_bytes)
                decoded_tensor = torch.from_numpy(decoded).float() / 255.0

            # Store result (CHW layout)
            result[i] = decoded_tensor.permute(2, 0, 1).to(device)

        return result

    def compress(self, x):
        """Compress images to JPEG byte buffers

        Args:
            x (torch.Tensor): Input batch of images [B,C,H,W]

        Returns:
            list: JPEG encoded byte buffers
        """
        if not HAS_NVJPEG:
            return self.cpu_fallback.compress(x)

        batch_size = x.shape[0]
        device = x.device
        jpeg_buffers = []

        for i in range(batch_size):
            # Convert to format expected by nvJPEG
            img = x[i].permute(1, 2, 0).mul(255).byte()

            # Encode using nvJPEG
            if device.type == 'cuda':
                img_cp = cp.asarray(img.cpu().numpy())
                jpeg_bytes = self.encoder.encode(img_cp)
            else:
                img_np = img.numpy()
                jpeg_bytes = self.encoder.encode(img_np)

            jpeg_buffers.append(jpeg_bytes)

        return jpeg_buffers

    def decompress(self, jpeg_buffers, device=None):
        """Decompress JPEG byte buffers

        Args:
            jpeg_buffers (list): List of JPEG encoded byte buffers
            device (torch.device): Device to place decoded tensors

        Returns:
            torch.Tensor: Batch of decompressed images [B,C,H,W]
        """
        if not HAS_NVJPEG:
            return self.cpu_fallback.decompress(jpeg_buffers, device)

        batch_size = len(jpeg_buffers)
        result_list = []

        for i in range(batch_size):
            # Decode JPEG bytes
            decoded = self.decoder.decode(jpeg_buffers[i])

            # Convert to torch tensor
            if device and device.type == 'cuda':
                decoded_tensor = torch.from_numpy(cp.asnumpy(decoded)).float() / 255.0
            else:
                decoded_tensor = torch.from_numpy(decoded).float() / 255.0

            # Channels last to channels first
            decoded_tensor = decoded_tensor.permute(2, 0, 1)
            result_list.append(decoded_tensor)

        # Stack batch
        result = torch.stack(result_list)
        if device:
            result = result.to(device)

        return result

# Create an example to check if the model works in only in gpu
if __name__ == "__main__":
    import torch
    from torchvision import transforms
    from PIL import Image

    # Load an example image
    img = Image.open("data/kodak/kodim01.png").convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    img_tensor = transform(img).unsqueeze(0).cuda()  # Add batch dimension and move to GPU

    # Create the model
    model = NvJPEGCompression().cuda()  # Move model to GPU

    # Forward pass
    output = model(img_tensor)

    print(output.shape)  # Should print the shape of the reconstructed image tensor