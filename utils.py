import os
import glob
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import yaml
from tqdm import tqdm

def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def preprocess(image: Image.Image, image_size: int) -> torch.Tensor:
    img = image.resize((image_size, image_size), Image.BILINEAR)
    img_np = np.array(img).astype(np.float32)
    img_np = (img_np / 127.5) - 1.0
    img_np = img_np.transpose(2, 0, 1)  # HWC -> CHW
    return torch.from_numpy(img_np)

class CartoonDataset(Dataset):
    def __init__(self, config):
        self.root_dir = config["data"]["dataset_path"]
        self.image_size = config["data"]["image_size"]
        self.target_attributes = config["data"]["attributes"]
        self.index_file = config["data"].get("index_file", "dataset_index.csv")
        self.subsec_count = int(config["data"].get("subsec_count", 10))
        self.scan_limit_per_folder = int(config["data"].get("scan_limit_per_folder", 500))

        # Force rescan each run (your current behavior)
        if os.path.exists(self.index_file):
            print("Removing old index file to force rescan...")
            os.remove(self.index_file)

        print(f"Indexing dataset at: {self.root_dir}")
        self.df = self._create_index()

        if len(self.df) > 0:
            self.df.to_csv(self.index_file, index=False)
            print(f"Index saved to {self.index_file} with {len(self.df)} images.")
        else:
            raise ValueError(f"Dataset is empty! Checked folders 0..{self.subsec_count-1} in {self.root_dir}.")

        # Map attributes to class indices
        self.attr_maps = {}
        for attr in self.target_attributes:
            if attr in self.df.columns:
                unique_vals = sorted(self.df[attr].unique())
                self.attr_maps[attr] = {val: i for i, val in enumerate(unique_vals)}
                print(f"Mapped '{attr}': {len(unique_vals)} classes")
            else:
                print(f"Warning: Attribute '{attr}' not found in index columns.")

    def _create_index(self):
        data = []
        for i in range(self.subsec_count):
            sub_folder_path = os.path.join(self.root_dir, str(i))
            print(f"Scanning folder: {sub_folder_path}")

            if not os.path.exists(sub_folder_path):
                print(f"   Folder not found: {sub_folder_path}")
                continue

            csv_files = glob.glob(os.path.join(sub_folder_path, "*.csv"))
            print(f"   Found {len(csv_files)} csv files.")

            if len(csv_files) == 0:
                continue

            take = min(len(csv_files), self.scan_limit_per_folder)
            for csv_file in tqdm(csv_files[:take], desc=f"Parsing Folder {i}", leave=True):
                try:
                    tmp_df = pd.read_csv(
                        csv_file,
                        header=None,
                        index_col=0,
                        on_bad_lines="skip"
                    )

                    row_data = {}
                    base_name = os.path.basename(csv_file)
                    img_name = base_name.replace(".csv", ".png")
                    row_data["rel_path"] = os.path.join(str(i), img_name)

                    for attr in self.target_attributes:
                        if attr in tmp_df.index:
                            # CSV format: "attr", value, cardinality
                            # tmp_df.loc[attr].values[0] => value
                            val = tmp_df.loc[attr].values[0]
                            try:
                                val = int(val)
                            except:
                                pass
                            row_data[attr] = val
                        else:
                            row_data[attr] = 0

                    data.append(row_data)
                except Exception:
                    continue

        return pd.DataFrame(data)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.root_dir, row["rel_path"])

        try:
            image = Image.open(img_path).convert("RGB")
            image = preprocess(image, self.image_size)
        except:
            image = torch.zeros(3, self.image_size, self.image_size)

        attr_vec = []
        for attr in self.target_attributes:
            if attr in self.attr_maps:
                val = row[attr]
                # map raw value -> class index
                attr_vec.append(self.attr_maps[attr].get(val, 0))
            else:
                attr_vec.append(0)

        return image, torch.tensor(attr_vec, dtype=torch.long)

def get_dataloader(config):
    dataset = CartoonDataset(config)

    num_workers = int(config["data"].get("num_workers", 2))
    # Colab-safe loader settings
    loader = DataLoader(
        dataset,
        batch_size=int(config["data"]["batch_size"]),
        shuffle=True,
        num_workers=2,  # force exactly 2
        pin_memory=True,
        drop_last=True,
        persistent_workers=True if num_workers > 0 else False,
        multiprocessing_context="spawn" if num_workers > 0 else None,
        prefetch_factor=2 if num_workers > 0 else None,
    )
    return loader
