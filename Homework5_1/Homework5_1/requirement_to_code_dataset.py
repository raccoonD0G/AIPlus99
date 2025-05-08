import json
from torch.utils.data import Dataset, DataLoader

class RequirementHeaderCppDataset(Dataset):
    def __init__(self, file_path, max_samples=None):
        self.data = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if max_samples and len(self.data) >= max_samples:
                    break
                example = json.loads(line)
                if "requirement" in example and "header_code" in example and "cpp_code" in example:
                    self.data.append({
                        "requirement": example["requirement"],
                        "header_code": example["header_code"],
                        "cpp_code": example["cpp_code"]
                    })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def get_train_dataloader(file_path="requirement_code_pairs.jsonl", batch_size=8, shuffle=True, limit=None):
    dataset = RequirementHeaderCppDataset(file_path, max_samples=limit)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        collate_fn=lambda batch: {
            "requirement":    [item["requirement"] for item in batch],
            "header_code":    [item["header_code"] for item in batch],
            "cpp_code":       [item["cpp_code"] for item in batch],
        }
    )
    return dataloader
