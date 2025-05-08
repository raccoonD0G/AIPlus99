import json
from torch.utils.data import Dataset, DataLoader

class CodeFormattingDataset(Dataset):
    def __init__(self, file_path):
        self.data = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                self.data.append({
                    "bad_code": obj["bad_code"],
                    "good_code": obj["good_code"]
                })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {
            "bad_code": self.data[idx]["bad_code"],
            "good_code": self.data[idx]["good_code"]
        }

def get_train_dataloader(file_path="code_formatting_pairs.jsonl", batch_size=8, shuffle=True):
    dataset = CodeFormattingDataset(file_path)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=True  # 마지막 배치 버려서 항상 일정한 batch_size 유지
    )
    return dataloader
