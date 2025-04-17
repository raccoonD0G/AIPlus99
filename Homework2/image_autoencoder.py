import os
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from tqdm import tqdm
import torch
import torch.nn as nn
from torchvision.datasets import CIFAR100

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
    def __init__(self, embedding_dim=768, weight_path="trained_image_autoencoder"):
        super().__init__()

        # Encoder 구조
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1),
            nn.Dropout2d(0.2),

            ResidualCNNBlock(32, dropout=0.2),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            nn.Dropout2d(0.2),

            ResidualCNNBlock(64, dropout=0.2),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),
            nn.Dropout2d(0.2),

            ResidualCNNBlock(128, dropout=0.2),

            nn.AdaptiveAvgPool2d((4, 4)),
        )

        self.projector = nn.Sequential(
            nn.Linear(2048, 512),
            nn.LayerNorm(512),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),

            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),

            nn.Linear(512, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),
        )

        # 가중치 로드
        encoder_path = os.path.join(weight_path, "encoder.pt")
        projector_path = os.path.join(weight_path, "projector.pt")

        if os.path.exists(encoder_path) and os.path.exists(projector_path):
            self.encoder.load_state_dict(torch.load(encoder_path, map_location="cpu"))
            self.projector.load_state_dict(torch.load(projector_path, map_location="cpu"))
            print(f"가중치를 받아옵니다 '{weight_path}'")
        else:
            print(f"가중치가 없습니다. 처음부터 학습합니다")

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        z = self.projector(x)
        return z  # (B, 768)


class Autoencoder(nn.Module):
    def __init__(self, embedding_dim=768):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1),
            nn.Dropout2d(0.2),

            ResidualCNNBlock(32, dropout=0.2),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            nn.Dropout2d(0.2),

            ResidualCNNBlock(64, dropout=0.2),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),
            nn.Dropout2d(0.2),

            ResidualCNNBlock(128, dropout=0.2),

            nn.AdaptiveAvgPool2d((4, 4)),
        )

        self.projector = nn.Sequential(
            nn.Linear(2048, 512),
            nn.LayerNorm(512),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),

            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),

            nn.Linear(512, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),
        )

        self.decoder = nn.Sequential(
            nn.Linear(768, 128 * 28 * 28),
            nn.LayerNorm(128 * 28 * 28),
            nn.LeakyReLU(0.1),
            nn.Unflatten(1, (128, 28, 28)),

            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),

            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1),

            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(32, 3, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        z = self.projector(x)
        out = self.decoder(z)
        return out, z

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
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

if __name__ == "__main__":
    dataset = CIFAR100(root=".", download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    model = Autoencoder().to("cuda")
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Training loop
    n_epochs = 10
    model.train()
    for epoch in range(n_epochs):
        total_loss = 0
        for imgs, targets in tqdm(dataloader, desc=f"Epoch {epoch+1}/{n_epochs}"):
            imgs, targets = imgs.to("cuda"), targets.to("cuda")
            optimizer.zero_grad()
            outputs, _ = model(imgs)
            loss = loss_fn(outputs, imgs)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        tqdm.write(f"Epoch {epoch+1} Loss: {total_loss:.4f}")

    os.makedirs("trained_image_autoencoder", exist_ok=True)
    torch.save(model.encoder.state_dict(), "trained_image_autoencoder/encoder.pt")
    torch.save(model.projector.state_dict(), "trained_image_autoencoder/projector.pt")
