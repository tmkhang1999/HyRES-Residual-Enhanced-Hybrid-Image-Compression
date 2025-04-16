import os
import shutil

# Path to your train directory
train_dir = './data/train'

# Loop through all subdirectories (classes)
for class_folder in os.listdir(train_dir):
    class_path = os.path.join(train_dir, class_folder)

    if os.path.isdir(class_path):
        for img_file in os.listdir(class_path):
            src_path = os.path.join(class_path, img_file)
            dst_path = os.path.join(train_dir, img_file)

            # Rename if there's a name conflict
            if os.path.exists(dst_path):
                base, ext = os.path.splitext(img_file)
                count = 1
                while os.path.exists(dst_path):
                    dst_path = os.path.join(train_dir, f"{base}_{count}{ext}")
                    count += 1

            shutil.move(src_path, dst_path)

        # Optionally remove the now-empty class folder
        os.rmdir(class_path)

print("All images moved to ./data/train")
