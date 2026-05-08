import os
import shutil

# ===== PATHS =====
DATASET_1 = "converted_dataset3"
DATASET_2 = "merged_database"

OUTPUT = "new_merged_dataset"

SPLITS = ["train", "valid", "test"]

# ===== CREATE OUTPUT STRUCTURE =====
for split in SPLITS:
    os.makedirs(os.path.join(OUTPUT, split, "images"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT, split, "labels"), exist_ok=True)


# ===== COPY FUNCTION =====
def copy_dataset(src_root, prefix):
    for split in SPLITS:

        img_src = os.path.join(src_root, split, "images")
        lbl_src = os.path.join(src_root, split, "labels")

        img_dst = os.path.join(OUTPUT, split, "images")
        lbl_dst = os.path.join(OUTPUT, split, "labels")

        if not os.path.exists(img_src):
            continue

        files = [f for f in os.listdir(img_src) if f.endswith((".jpg", ".png"))]

        for file in files:

            src_img_path = os.path.join(img_src, file)
            lbl_name = file.replace(".jpg", ".txt").replace(".png", ".txt")
            src_lbl_path = os.path.join(lbl_src, lbl_name)

            # 🚫 skip if label missing
            if not os.path.exists(src_lbl_path):
                continue

            # ===== rename to avoid collisions =====
            new_name = f"{prefix}_{file}"
            new_lbl = new_name.replace(".jpg", ".txt").replace(".png", ".txt")

            # ===== copy =====
            shutil.copy(src_img_path, os.path.join(img_dst, new_name))
            shutil.copy(src_lbl_path, os.path.join(lbl_dst, new_lbl))


# ===== RUN =====
copy_dataset(DATASET_1, "ds1")
copy_dataset(DATASET_2, "ds2")

print("✅ Datasets merged successfully")