import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from torchvision import transforms
from torchvision import utils as vutils
from PIL import Image
import requests
from io import BytesIO

from transformers import Trainer, TrainingArguments, TrainerCallback
from datasets import load_dataset
from tqdm import tqdm



class SaveAutoencoderWeightsCallback(TrainerCallback):
    def __init__(self, dataset, save_dir="trained_image_autoencoder"):
        self.dataset = dataset
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        if model is not None:
            encoder_path = os.path.join(self.save_dir, f"encoder_epoch_{state.epoch:.0f}.pt")
            projector_path = os.path.join(self.save_dir, f"projector_epoch_{state.epoch:.0f}.pt")
            torch.save(model.encoder.state_dict(), encoder_path)
            torch.save(model.projector.state_dict(), projector_path)
            tqdm.write(f"에포크 {state.epoch:.0f} - 가중치 저장 완료")

            # Trainer 학습 후 실행
            save_decoded_samples(model, self.dataset)


class ResidualCNNBlock(nn.Module):
    def __init__(self, channels, dropout=0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.LeakyReLU(0.1),
            nn.Dropout2d(dropout),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
        )
        self.activation = nn.LeakyReLU(0.1)

    def forward(self, x):
        return self.activation(x + self.block(x))


class ImageEncoder(nn.Module):
    def __init__(self, embedding_dim=1024, weight_path="trained_image_autoencoder", device="cuda"):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),   # 512 → 256
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            nn.Dropout2d(0.1),
            ResidualCNNBlock(64),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # 256 → 128
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),
            nn.Dropout2d(0.1),
            ResidualCNNBlock(128),

            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),  # 128 → 64
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),
            nn.Dropout2d(0.1),
            ResidualCNNBlock(256),

            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),  # 64 → 32
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1),
            nn.Dropout2d(0.1),
            ResidualCNNBlock(512),

            nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1),  # 32 → 16
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1),
            nn.Dropout2d(0.1),

            nn.AdaptiveAvgPool2d((4, 4))  # → (1024, 4, 4)
        )

        self.projector = nn.Sequential(
            nn.Flatten(),  # 1024 * 4 * 4 = 16384
            nn.Linear(16384, 4096),
            nn.LayerNorm(4096),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),

            nn.Linear(4096, 2048),
            nn.LayerNorm(2048),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3), 

            nn.Linear(2048, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3), 
        )

        self._try_load_weights(weight_path, device)

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        return self.projector(x)

    def _try_load_weights(self, weight_path, device):
        if not os.path.exists(weight_path):
            tqdm.write(f"'{weight_path}' 경로가 존재하지 않음. 가중치를 로드하지 않음.")
            return

        epochs = [
            int(f.split("_")[-1].split(".")[0])
            for f in os.listdir(weight_path)
            if f.startswith("encoder_epoch_") and f.endswith(".pt")
        ]
        if not epochs:
            tqdm.write(f"'{weight_path}'에 저장된 에포크 기반 가중치가 없음.")
            return

        epoch = max(epochs)
        encoder_path = os.path.join(weight_path, f"encoder_epoch_{epoch}.pt")
        projector_path = os.path.join(weight_path, f"projector_epoch_{epoch}.pt")

        try:
            self.encoder.load_state_dict(torch.load(encoder_path, map_location=device))
            self.projector.load_state_dict(torch.load(projector_path, map_location=device))
            tqdm.write(f"에포크 {epoch} 기준 가중치 자동 로드 완료 from '{weight_path}' : ImageEncoder")
        except Exception as e:
            tqdm.write(f"가중치 로드 실패: {e}")


import torch
import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self, embedding_dim=1024, weight_path="trained_image_autoencoder", device="cuda"):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),   # 512 → 256
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            nn.Dropout2d(0.1),
            ResidualCNNBlock(64),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # 256 → 128
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),
            nn.Dropout2d(0.1),
            ResidualCNNBlock(128),

            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),  # 128 → 64
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),
            nn.Dropout2d(0.1),
            ResidualCNNBlock(256),

            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),  # 64 → 32
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1),
            nn.Dropout2d(0.1),
            ResidualCNNBlock(512),

            nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1),  # 32 → 16
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1),
            nn.Dropout2d(0.1),

            nn.AdaptiveAvgPool2d((4, 4))  # → (1024, 4, 4)
        )

        self.projector = nn.Sequential(
            nn.Flatten(),  # 1024 * 4 * 4 = 16384
            nn.Linear(16384, 4096),
            nn.LayerNorm(4096),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),

            nn.Linear(4096, 2048),
            nn.LayerNorm(2048),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3), 

            nn.Linear(2048, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3), 
        )

        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, 512 * 8 * 8),
            nn.LayerNorm(512 * 8 * 8),
            nn.LeakyReLU(0.1),
            nn.Unflatten(1, (512, 8, 8)),

            nn.Upsample(scale_factor=2),  # 8 → 16
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),

            nn.Upsample(scale_factor=2),  # 16 → 32
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),

            nn.Upsample(scale_factor=2),  # 32 → 64
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),

            nn.Upsample(scale_factor=2),  # 64 → 128
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1),

            nn.Upsample(scale_factor=2),  # 128 → 256
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.1),

            nn.Upsample(scale_factor=2),  # 256 → 512
            nn.Conv2d(16, 3, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

        self._try_load_weights(weight_path, device)

    def forward(self, input=None, labels=None):
        x = self.encoder(input)
        x = x.view(x.size(0), -1)
        z = self.projector(x)
        recon = self.decoder(z)

        loss = None
        if labels is not None:
            loss = nn.functional.mse_loss(recon, labels, reduction="mean")

        return {"loss": loss, "logits": recon}

    def _try_load_weights(self, weight_path, device="cpu"):
        if not os.path.exists(weight_path):
            tqdm.write(f"'{weight_path}' 경로가 존재하지 않음. 가중치를 로드하지 않음.")
            return

        epochs = [
            int(f.split("_")[-1].split(".")[0])
            for f in os.listdir(weight_path)
            if f.startswith("encoder_epoch_") and f.endswith(".pt")
        ]
        if not epochs:
            tqdm.write(f"'{weight_path}'에 저장된 가중치가 없음. 새로 학습합니다.")
            return

        epoch = max(epochs)
        encoder_path = os.path.join(weight_path, f"encoder_epoch_{epoch}.pt")
        projector_path = os.path.join(weight_path, f"projector_epoch_{epoch}.pt")

        try:
            self.encoder.load_state_dict(torch.load(encoder_path, map_location=device))
            self.projector.load_state_dict(torch.load(projector_path, map_location=device))
            tqdm.write(f"에포크 {epoch} 기준 가중치 자동 로드 완료 from '{weight_path}'")
        except Exception as e:
            tqdm.write(f"가중치 로드 실패: {e}")



class ImageFolderDataset(Dataset):
    def __init__(self, folder, transform=None):
        self.paths = [os.path.join(folder, fname) for fname in os.listdir(folder) if fname.endswith('.png')]
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        image = Image.open(self.paths[idx]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, image


transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor()
])

# 샘플 저장 함수
def save_decoded_samples(model, dataset, device="cuda", max_samples=16):
    model.eval()
    with torch.no_grad():
        for i in range(min(max_samples, len(dataset))):
            input_img = dataset[i]["input"].unsqueeze(0).to(device)
            recon = model(input=input_img)["logits"].cpu().squeeze(0)
            path = os.path.join(output_dir, f"sample_{i:03d}.png")
            vutils.save_image(recon, path)
            print(f"저장 완료: {path}")


from torch.utils.data import Dataset
from PIL import Image
import os

class ImageFolderDataset(Dataset):
    def __init__(self, folder, transform=None):
        self.paths = [os.path.join(folder, fname)
                      for fname in os.listdir(folder)
                      if fname.lower().endswith(('.png', '.jpg', '.jpeg'))]
        self.transform = transform or transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        img = self.transform(img)
        return {"input": img, "labels": img}


if __name__ == "__main__":
    from transformers import Trainer, TrainingArguments

    image_folder = r"C:\Users\Monstera\Desktop\AI\Homework3_0_0\archive\Images"
    dataset = ImageFolderDataset(image_folder)

    def data_collator(batch):
        inputs = torch.stack([item["input"] for item in batch])
        labels = torch.stack([item["labels"] for item in batch])
        return {"input": inputs, "labels": labels}

    model = Autoencoder().to("cuda")

    training_args = TrainingArguments(
        output_dir="./autoencoder_trainer_output",
        per_device_train_batch_size=16,
        num_train_epochs=10,
        logging_dir="./logs",
        logging_steps=10,
        save_strategy="epoch",
        remove_unused_columns=False,
    )

    output_dir = "image_autoencoder_output"
    os.makedirs(output_dir, exist_ok=True)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
        callbacks=[SaveAutoencoderWeightsCallback(dataset, save_dir="trained_image_autoencoder")]
    )

    trainer.train()
    save_decoded_samples(model, dataset)
