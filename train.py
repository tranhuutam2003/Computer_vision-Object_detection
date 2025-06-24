import os
import csv
import random

def generate_train_val(label_csv, image_dir, train_txt, val_txt, val_ratio=0.1, seed=42):
    """
    Args:
        label_csv (str): CSV file with two columns: filename,label
        image_dir (str): Directory where images are stored
        train_txt (str): Output path for train.txt
        val_txt (str): Output path for val.txt
        val_ratio (float): Fraction of data for validation
        seed (int): Random seed for shuffling
    """
    # Read label CSV
    with open(label_csv, newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        entries = [(row[0], row[1]) for row in reader]

    # Shuffle and split
    random.seed(seed)
    random.shuffle(entries)
    split_idx = int(len(entries) * (1 - val_ratio))
    train_entries = entries[:split_idx]
    val_entries = entries[split_idx:]

    # Write train.txt
    with open(train_txt, 'w', encoding='utf-8') as f_train:
        for img_name, label in train_entries:
            f_train.write(f"{os.path.join(image_dir, img_name)}\t{label}\n")

    # Write val.txt
    with open(val_txt, 'w', encoding='utf-8') as f_val:
        for img_name, label in val_entries:
            f_val.write(f"{os.path.join(image_dir, img_name)}\t{label}\n")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate train/val txt for recognition.")
    parser.add_argument("--csv", default='../demo/rec_plate.csv', help="CSV with filename,label")
    parser.add_argument("--img_dir", default='../dataset/plate', help="Directory of cropped images")
    parser.add_argument("--train_txt", default="../dataset/plate/rec/train.txt", help="Output train.txt")
    parser.add_argument("--val_txt", default="../dataset/plate/rec/val.txt", help="Output val.txt")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Validation split ratio")
    args = parser.parse_args()
    generate_train_val(args.csv, args.img_dir, args.train_txt, args.val_txt, args.val_ratio)