import io
import torch
from torch import nn
from torchvision import transforms
from turbojpeg import TurboJPEG


class TurboJPEGCompression(nn.Module):
    def __init__(self, quality=10):
        super().__init__()
        self.quality = quality
        self.jpeg = TurboJPEG()
        self.to_pil = transforms.ToPILImage()
        self.to_tensor = transforms.ToTensor()
        print(f"Using TurboJPEG compression with quality {quality}")

    def compress(self, x):
        # Always work on CPU for JPEG operations
        x_cpu = x.cpu() if x.device.type != 'cpu' else x
        batch_size = x_cpu.size(0)
        compressed_buffers = []

        for i in range(batch_size):
            # Handle both RGB and grayscale images
            img_tensor = torch.clamp(x_cpu[i], 0, 1)

            # Check if image is grayscale (1 channel) and convert to RGB if needed
            if img_tensor.size(0) == 1:
                img_tensor = img_tensor.repeat(3, 1, 1)

            # Convert tensor to numpy array directly
            img_np = (img_tensor.permute(1, 2, 0) * 255).byte().numpy()

            # Compress with TurboJPEG (much faster than PIL)
            jpeg_data = self.jpeg.encode(img_np, quality=self.quality)

            # Store in buffer
            buffer = io.BytesIO(jpeg_data)
            compressed_buffers.append(buffer)

        return compressed_buffers

    def decompress(self, compressed_buffers, device):
        batch_size = len(compressed_buffers)
        decompressed_images = []

        for i in range(batch_size):
            # Get bytes from buffer
            buffer_bytes = compressed_buffers[i].getvalue()

            # Decompress JPEG using TurboJPEG
            decoded_img = self.jpeg.decode(buffer_bytes)

            # Convert to tensor and normalize to [0,1]
            tensor_img = torch.from_numpy(decoded_img).float().permute(2, 0, 1) / 255.0
            tensor_img = tensor_img.to(device)

            decompressed_images.append(tensor_img)

        return torch.stack(decompressed_images, dim=0)

    def forward(self, x):
        # Get the device for later use
        device = x.device

        # Process on CPU
        compressed_buffers = self.compress(x)

        # Calculate bits per pixel
        N, _, H, W = x.size()
        num_pixels = N * H * W
        compressed_bits = sum(len(buffer.getvalue()) * 8 for buffer in compressed_buffers)
        jpeg_bpp = compressed_bits / num_pixels

        # Return to original device
        decompressed = self.decompress(compressed_buffers, device)
        return decompressed, jpeg_bpp


if __name__ == "__main__":
    import os
    import torch
    from torch import nn
    from torchvision import transforms
    from PIL import Image
    from models.utils.turbo_jpeg_compression import TurboJPEGCompression

    # Path to test images
    # Kodak images shipped with the repo: <repo>/data/test
    test_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "test")

    # Set quality levels to test
    quality_levels = [1, 10, 25, 50, 75, 95]

    for quality in quality_levels:
        print(f"Testing JPEG quality level: {quality}")
        jpeg_compressor = TurboJPEGCompression(quality=quality)

        total_bpp = 0
        total_mse = 0
        total_psnr = 0
        image_count = 0

        # Process all images in the test directory
        for filename in os.listdir(test_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                image_path = os.path.join(test_dir, filename)

                try:
                    # Load image and convert to tensor
                    image = Image.open(image_path).convert("RGB")
                    image_tensor = transforms.ToTensor()(image).unsqueeze(0)

                    # Compress
                    compressed_data = jpeg_compressor.compress(image_tensor)
                    compressed_size = sum(len(buffer.getvalue()) for buffer in compressed_data)
                    _, c, h, w = image_tensor.shape
                    num_pixels = h * w
                    bpp = compressed_size * 8 / num_pixels

                    # Decompress
                    decompressed_tensor = jpeg_compressor.decompress(compressed_data, device=torch.device("cpu"))

                    # Calculate MSE loss (scaled to 0-255 range)
                    mse_loss = nn.MSELoss(reduction='mean')
                    mse = mse_loss(image_tensor, decompressed_tensor) * 255 ** 2
                    psnr = 10 * torch.log10(255 ** 2 / mse) if mse > 0 else torch.tensor(float('inf'))

                    # Accumulate metrics
                    total_bpp += bpp
                    total_mse += mse.item()
                    total_psnr += psnr.item()
                    image_count += 1

                    print(f"  {filename}: Bpp: {bpp:.4f}, MSE: {mse.item():.4f}, PSNR: {psnr.item():.2f} dB")

                except Exception as e:
                    print(f"  Error processing {filename}: {e}")

        # Calculate and display averages
        if image_count > 0:
            avg_bpp = total_bpp / image_count
            avg_mse = total_mse / image_count
            print(f"\nQuality {quality} - Average Bpp: {avg_bpp:.4f}, Average MSE: {avg_mse:.4f}, Average PSNR: {total_psnr / image_count:.2f} dB")
            print("=" * 80)