import tensorflow as tf
import torch


class TFJPEGCompression:
    def __init__(self, quality=75):
        self.quality = quality

    def __call__(self, x):
        # Encode and decode
        encoded = [tf.io.encode_jpeg(img, quality=self.quality) for img in x]
        decoded = [tf.io.decode_jpeg(img) for img in encoded]

        # Back to torch format [B,C,H,W] in [0,1]
        result = tf.stack(decoded)
        result = torch.from_numpy(result.numpy()).permute(0, 3, 1, 2) / 255.0
        return result.to(x.device)


# Test the TFJPEGCompression class
if __name__ == "__main__":
    # Create a random tensor simulating an image batch
    batch_size = 4
    channels = 3
    height = 256
    width = 256
    x = torch.rand(batch_size, height, width, channels)

    # Initialize the TFJPEGCompression class
    jpeg_compressor = TFJPEGCompression(quality=75)

    # Compress and decompress the image batch
    compressed_images = jpeg_compressor(x)

    # Print the shape of the compressed images
    print(compressed_images.shape)