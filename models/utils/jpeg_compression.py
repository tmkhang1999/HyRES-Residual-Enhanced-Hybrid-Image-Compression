import io
import torch
from PIL import Image
from torch import nn
from torchvision import transforms


class JPEGCompression(nn.Module):
    def __init__(self, quality=25):
        super().__init__()
        self.quality = quality
        self.to_pil = transforms.ToPILImage()
        self.to_tensor = transforms.ToTensor()

    def compress(self, x):
        batch_size = x.size(0)
        compressed_buffers = []

        for i in range(batch_size):
            # Handle both RGB and grayscale images
            img_tensor = torch.clamp(x[i], 0, 1)

            # Check if image is grayscale (1 channel) and convert to RGB if needed
            if img_tensor.size(0) == 1:
                img_tensor = img_tensor.repeat(3, 1, 1)

            # Convert to PIL image
            pil_img = self.to_pil(img_tensor)

            # Ensure RGB mode
            if pil_img.mode != "RGB":
                pil_img = pil_img.convert("RGB")

            # Compress with JPEG
            buffer = io.BytesIO()
            pil_img.save(buffer, format='JPEG', quality=self.quality)
            buffer.seek(0)

            compressed_buffers.append(buffer)

        return compressed_buffers

    def decompress(self, compressed_buffers, device):
        batch_size = len(compressed_buffers)
        decompressed_images = []

        for i in range(batch_size):
            # Reset buffer position
            compressed_buffers[i].seek(0)

            # Decompress JPEG
            pil_img = Image.open(compressed_buffers[i])

            # Ensure we're working with RGB
            if pil_img.mode != "RGB":
                pil_img = pil_img.convert("RGB")

            tensor_img = self.to_tensor(pil_img).to(device)
            decompressed_images.append(tensor_img)

        return torch.stack(decompressed_images, dim=0)

    def forward(self, x):
        compressed = self.compress(x)
        decompressed = self.decompress(compressed, x.device)
        return decompressed


if __name__ == "__main__":
    jpeg_compressor = JPEGCompression(quality=1)

    # Load image and ensure it's RGB
    image = Image.open("/Users/khangtran/Documents/Programming/Research/HyRES/data/kodak/kodim01.png").convert("RGB")

    # Convert to tensor and add batch dimension
    image_tensor = transforms.ToTensor()(image).unsqueeze(0)

    # Compress and decompress
    compressed_data = jpeg_compressor.compress(image_tensor)

    # Calculate bpp correctly
    compressed_size = sum(len(buffer.getvalue()) for buffer in compressed_data)
    _, c, h, w = image_tensor.shape
    num_pixels = h * w
    bpp = compressed_size * 8 / num_pixels
    print(f"Image size: {num_pixels} pixels")
    print(f"Compressed size: {compressed_size} bytes")
    print(f"Bits per pixel: {bpp:.4f}")

    decompressed_tensor = jpeg_compressor.decompress(compressed_data, device=torch.device("cpu"))

    # Check tensor shape and values
    print(f"Decompressed tensor shape: {decompressed_tensor.shape}")

    decompressed_image = transforms.ToPILImage()(decompressed_tensor[0])

    # Show the residual map between original and decompressed image in tensor
    residual_map = image_tensor[0] - decompressed_tensor[0]
    # residual_map = residual_map * 0.5 + 0.5
    residual_map = transforms.ToPILImage()(residual_map.cpu())
    residual_map.show()

    # x_hat = decompressed_tensor[0] + residual_map
    # x_hat = transforms.ToPILImage()(x_hat.cpu())
    # x_hat.show()


    # Save for comparison
    image.save("original.png")
    decompressed_image.save("decompressed.png")

    # decompressed_image.show()