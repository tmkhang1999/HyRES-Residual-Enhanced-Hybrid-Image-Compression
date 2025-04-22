import io
import torch
from torch import nn
from torchvision import transforms
from turbojpeg import TurboJPEG


class TurboJPEGCompression(nn.Module):
    def __init__(self, quality=25):
        super().__init__()
        self.quality = quality
        self.jpeg = TurboJPEG(lib_path='/home/tm05393z/miniconda3/envs/myenv/lib/libturbojpeg.so')
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
    jpeg_compressor = TurboJPEGCompression(quality=25)

    # Load image and ensure it's RGB
    from PIL import Image
    image = Image.open("/Users/khangtran/Documents/Programming/Research/HyRES/data/train/n02971356_5846.JPEG").convert("RGB")
    image_tensor = transforms.ToTensor()(image).unsqueeze(0)

    # Compress
    compressed_data = jpeg_compressor.compress(image_tensor)
    compressed_size = sum(len(buffer.getvalue()) for buffer in compressed_data)
    _, c, h, w = image_tensor.shape
    num_pixels = h * w
    bpp = compressed_size * 8 / num_pixels
    print(f"Image size: {num_pixels} pixels")
    print(f"Compressed size: {compressed_size} bytes")
    print(f"Bits per pixel: {bpp:.4f}")

    # Decompress
    decompressed_tensor = jpeg_compressor.decompress(compressed_data, device=torch.device("cpu"))
    print(f"Decompressed tensor shape: {decompressed_tensor.shape}")
    # decompressed_image = transforms.ToPILImage()(decompressed_tensor[0])
    # decompressed_image.show()

    # Show the residual map between original and decompressed image in tensor
    residual_map = image_tensor[0] - decompressed_tensor[0]
    # residual_map = transforms.ToPILImage()(residual_map.cpu())
    # residual_map.show()

    x_hat = decompressed_tensor[0] + residual_map
    # x_hat = torch.clamp(x_hat, 0, 1)
    x_hat = transforms.ToPILImage()(x_hat.cpu())
    x_hat.show()