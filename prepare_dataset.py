import os
import scipy.io
import cv2
from sklearn.model_selection import train_test_split
from pathlib import Path
import shutil
import random
from tqdm import tqdm

# Config
TARGET_SIZE = 640  # Reduced from original 1024+ sizes
MAX_SAMPLES = 2000  # Limit total samples
UOB_SAMPLE_RATIO = 0.3  # Only use 30% of UoB images


def convert_shanghaitech_to_yolo(base_path):
    """Convert ShanghaiTech with size filtering"""
    parts = ["part_A", "part_B"]
    all_files = []

    for part in parts:
        for split in ["train_data", "test_data"]:
            img_dir = base_path / part / split / "images"
            gt_dir = base_path / part / split / "ground-truth"

            for img_file in img_dir.glob("*.jpg"):
                mat_file = gt_dir / f"GT_{img_file.stem}.mat"
                if mat_file.exists():
                    all_files.append((img_file, mat_file, "shanghai"))

    return all_files


def convert_ucf_cc50_to_yolo(base_path):
    """Convert UCF_CC_50 with size filtering"""
    all_files = []
    for img_file in base_path.glob("*.jpg"):
        mat_file = base_path / f"{img_file.stem}_ann.mat"
        if mat_file.exists():
            all_files.append((img_file, mat_file, "ucf"))
    return all_files


def convert_uob_graduation_to_yolo(base_path):
    """Convert UoB with sampling and size filtering"""
    all_files = []
    img_dir = base_path / "UoB_Graduation_Ceremony_Day"

    if img_dir.exists():
        for img_file in img_dir.glob("*.jpg"):
            if random.random() < UOB_SAMPLE_RATIO:  # Random sampling
                all_files.append((img_file, None, "uob"))  # None for no .mat files

    return all_files


def process_and_resize(files, output_dir, split):
    """Process files with resizing and balanced sampling"""
    os.makedirs(output_dir / "images" / split, exist_ok=True)
    os.makedirs(output_dir / "labels" / split, exist_ok=True)

    for img_path, mat_file, dataset in tqdm(files, desc=f"Processing {split}"):
        # Load and resize
        img = cv2.imread(str(img_path))
        img = cv2.resize(img, (TARGET_SIZE, TARGET_SIZE))

        # Save resized image
        new_img_path = output_dir / "images" / split / img_path.name
        cv2.imwrite(str(new_img_path), img)

        # Handle labels
        label_path = output_dir / "labels" / split / f"{img_path.stem}.txt"

        if dataset in ["shanghai", "ucf"]:
            mat = scipy.io.loadmat(str(mat_file))
            points = (
                mat["image_info"][0][0][0][0][0]
                if dataset == "shanghai"
                else mat["annPoints"]
            )

            with open(label_path, "w") as f:
                for point in points:
                    x, y = (
                        point[0] * TARGET_SIZE / img.shape[1],
                        point[1] * TARGET_SIZE / img.shape[0],
                    )
                    f.write(
                        f"0 {x / TARGET_SIZE:.6f} {y / TARGET_SIZE:.6f} 0.01 0.01\n"
                    )

        elif dataset == "uob":
            # PLACEHOLDER - REPLACE WITH REAL ANNOTATIONS
            with open(label_path, "w") as f:
                # Dummy single person at center
                f.write("0 0.5 0.5 0.05 0.05\n")


if __name__ == "__main__":
    base_path = Path("data")
    output_dir = Path("data/yolo_optimized")

    # Clean and prepare
    if output_dir.exists():
        shutil.rmtree(output_dir)

    # Collect all files
    all_files = []
    all_files.extend(convert_shanghaitech_to_yolo(base_path / "ShanghaiTech"))
    all_files.extend(convert_ucf_cc50_to_yolo(base_path / "UCF_CC_50"))
    all_files.extend(convert_uob_graduation_to_yolo(base_path))

    # Sample if too large
    if len(all_files) > MAX_SAMPLES:
        all_files = random.sample(all_files, MAX_SAMPLES)

    # Split and process
    train_files, val_files = train_test_split(all_files, test_size=0.2, random_state=42)
    process_and_resize(train_files, output_dir, "train")
    process_and_resize(val_files, output_dir, "val")

    print(f"Created optimized dataset at {output_dir} with {len(all_files)} samples")
