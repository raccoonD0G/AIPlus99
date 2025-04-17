import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from PIL import Image
from io import BytesIO
import requests
import hashlib

from torchvision import transforms
from torchvision.utils import save_image

from tqdm import tqdm
from datasets import load_dataset
from diffusers import StableDiffusionPipeline
import matplotlib.pyplot as plt


from transformers import (
    DistilBertTokenizer,
    DistilBertModel,
    Trainer,
    TrainingArguments,
    TrainerCallback
)


class GANSaveCallback(TrainerCallback):
    def __init__(self, save_dir="trained_gan_generator", sample_dir=".", device="cuda"):
        self.save_dir = save_dir
        self.sample_dir = sample_dir
        self.device = device
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(sample_dir, exist_ok=True)

    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        epoch = int(state.epoch or 0)
        generator = model.generator
        text_encoder = model.text_encoder

        # 가중치 저장
        path = os.path.join(self.save_dir, f"encoder_epoch_{epoch}.pt")
        torch.save(generator.state_dict(), path)
        print(f"Generator 가중치 저장 완료: {path}")

        # 샘플 이미지 저장
        sample_text = "A cat sitting on a couch."
        generator.eval()
        text_encoder.eval()
        with torch.no_grad():
            text_embed = text_encoder([sample_text]).to(self.device)
            fake_img = generator(text_embed)[0].cpu()
            image_path = os.path.join(self.sample_dir, f"generated_sample_epoch{epoch}.png")
            save_image(fake_img, image_path)
            print(f"샘플 이미지 저장 완료: {image_path}")


def download_image_from_url(url):
    try:
        response = requests.get(url, timeout=3)
        img = Image.open(BytesIO(response.content))
        if img.mode == "P":
            img = img.convert("RGBA")
        img = img.convert("RGB")
        return img
    except:
        return None  
    
class TextEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
        
        for param in self.bert.parameters():
            param.requires_grad = False

    def forward(self, texts):
        tokens = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        tokens = {k: v.to("cuda") for k, v in tokens.items()}
        outputs = self.bert(**tokens)
        return outputs.last_hidden_state[:, 0, :] 


z_dim = 768


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, norm="group", dropout=False, num_groups=8):
        super().__init__()

        layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)]

        # 정규화 타입 선택
        if norm == "batch":
            layers.append(nn.BatchNorm2d(out_channels))
        elif norm == "group":
            layers.append(nn.GroupNorm(num_groups, out_channels))
        elif norm == "none":
            pass 

        layers.append(nn.LeakyReLU(0.1))

        if dropout:
            layers.append(nn.Dropout2d(0.1))

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)
    

class ResidualBlock(nn.Module):
    def __init__(self, channels, norm="group", dropout=False, num_groups=8):
        super().__init__()
        self.block = nn.Sequential(
            ConvBlock(channels, channels, norm=norm, dropout=dropout, num_groups=num_groups)
        )

    def forward(self, x):
        return x + self.block(x)


class Generator(nn.Module):
    def __init__(self, z_dim=768, norm="group", dropout=False, num_groups=8, weight_path="trained_gan_generator", device="cuda"):
        super().__init__()

        self.initial = nn.Sequential(
            nn.Linear(z_dim, 128 * 7 * 7),
            nn.LayerNorm(128 * 7 * 7),
            nn.LeakyReLU(0.1)
        )

        self.unflatten = nn.Unflatten(1, (128, 7, 7))

        def up_block(in_c, out_c):
            return nn.Sequential(
                nn.Upsample(scale_factor=2),
                ConvBlock(in_c, out_c, norm=norm, dropout=dropout, num_groups=num_groups),
                ResidualBlock(out_c, norm=norm, dropout=dropout, num_groups=num_groups)
            )

        self.upsample_blocks = nn.Sequential(
            up_block(128, 128),
            up_block(128, 128),
            up_block(128, 64), 
            up_block(64, 64),  
            up_block(64, 32),  
            up_block(32, 32),  
            nn.Sequential(     
                nn.Upsample(size=(512, 512)),
                ConvBlock(32, 32, norm=norm, dropout=dropout, num_groups=num_groups),
                ResidualBlock(32, norm=norm, dropout=dropout, num_groups=num_groups)
            )
        )

        self.output_layer = nn.Sequential(
            nn.Conv2d(32, 3, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

        # 사전 학습된 generator 가중치 로드
        self._try_load_weights(weight_path, device)
        

    def forward(self, x):
        x = self.initial(x)
        x = self.unflatten(x)
        x = self.upsample_blocks(x)
        return self.output_layer(x)
    
    def _try_load_weights(self, weight_path, device):
        if not os.path.exists(weight_path):
            tqdm.write(f"'{weight_path}' 경로가 존재하지 않음. Generator 가중치를 로드하지 않음.")
            return

        epochs = [
            int(f.split("_")[-1].split(".")[0])
            for f in os.listdir(weight_path)
            if f.startswith("encoder_epoch_") and f.endswith(".pt")
        ]
        if not epochs:
            tqdm.write(f"'{weight_path}'에 저장된 Generator 가중치가 없음.")
            return

        epoch = max(epochs)
        encoder_path = os.path.join(weight_path, f"encoder_epoch_{epoch}.pt")

        try:
            self.load_state_dict(torch.load(encoder_path, map_location=device))
            tqdm.write(f"에포크 {epoch} 기준 Generator 가중치 자동 로드 완료 from '{weight_path}'")
        except Exception as e:
            tqdm.write(f"Generator 가중치 로드 실패: {e}")




class Discriminator(nn.Module):
    def __init__(self, norm="group", dropout=True):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            ConvBlock(3, 64, norm=norm, dropout=dropout),
            nn.Conv2d(64, 64, 4, stride=2, padding=1),  
            nn.LeakyReLU(0.2),
            ResidualBlock(64, norm=norm, dropout=dropout),

            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            ResidualBlock(128, norm=norm, dropout=dropout),

            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            ResidualBlock(256, norm=norm, dropout=dropout),

            nn.Conv2d(256, 512, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            ResidualBlock(512, norm=norm, dropout=dropout),

            nn.AdaptiveAvgPool2d((7, 7))
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        return self.classifier(x)
        return self.classifier(x)


def compute_gradient_penalty(D, real_samples, fake_samples):
    alpha = torch.rand(real_samples.size(0), 1, 1, 1).to(real_samples.device)
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)

    d_interpolates = D(interpolates)
    fake = torch.ones(d_interpolates.size(), device=real_samples.device, requires_grad=False)

    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]

    gradients = gradients.reshape(gradients.size(0), -1)
    grad_norm = gradients.norm(2, dim=1)
    gp = ((grad_norm - 1) ** 2).mean()
    return gp

class LAIONDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform or transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        row = self.dataset[idx]
        text = row["caption"]
        img = download_image_from_url(row["url"])

        if img is None:
            # 다운로드 실패 시 dummy 반환 or 재시도
            return self.__getitem__((idx + 1) % len(self))

        return {
            "text": text,
            "image": self.transform(img)
        }




class GANModule(nn.Module):
    def __init__(self, generator, discriminator, text_encoder):
        super().__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.text_encoder = text_encoder

    def forward(self, batch):
        texts, real_imgs = batch
        text_embeds = self.text_encoder(texts)
        fake_imgs = self.generator(text_embeds)
        return {
            "real_imgs": real_imgs,
            "fake_imgs": fake_imgs,
            "text_embeds": text_embeds
        }


class GANTrainer(Trainer):
    def __init__(self, *args, gp_weight=10, **kwargs):
        super().__init__(*args, **kwargs)
        self.gp_weight = gp_weight
        self.opt_d = optim.Adam(self.model.discriminator.parameters(), lr=1e-5, betas=(0.0, 0.9))
        self.opt_g = optim.Adam(self.model.generator.parameters(), lr=1e-5, betas=(0.0, 0.9))

        self.step_counter = 0
        
    def compute_loss(self, model, inputs, return_outputs=False):
        batch_texts = inputs["text"]
        real_imgs = inputs["image"].to(self.args.device).requires_grad_(True)
        
        # DiscAriminator 학습용
        with torch.no_grad():
            text_embeds_d = self.model.text_encoder(batch_texts)
            fake_imgs_d = self.model.generator(text_embeds_d).detach()

        d_real = self.model.discriminator(real_imgs)
        d_fake = self.model.discriminator(fake_imgs_d)
        gp = compute_gradient_penalty(self.model.discriminator, real_imgs, fake_imgs_d)
        loss_d = -torch.mean(d_real) + torch.mean(d_fake) + self.gp_weight * gp

        # Generator 학습용
        text_embeds_g = self.model.text_encoder(batch_texts)
        fake_imgs_g = self.model.generator(text_embeds_g)

        d_fake_for_g = self.model.discriminator(fake_imgs_g)
        loss_g = -torch.mean(d_fake_for_g)

        total_loss = loss_d + loss_g
        
        self.log({
            "loss_d": loss_d.item(),
            "loss_g": loss_g.item(),
            "total_loss": total_loss.item(),
        })
        
        return total_loss, (loss_d, loss_g)

    def training_step(self, model, inputs, num_items_in_batch=None):
        model.train()
        self.step_counter += 1

        # Discriminator 학습
        total_loss, (loss_d, _) = self.compute_loss(model, inputs)
        self.opt_d.zero_grad()
        loss_d.backward()
        self.opt_d.step()

        # Generator는 5 스텝마다, 그리고 D가 충분히 잘할 때만 학습
        if self.step_counter % 5 == 0:
            batch_texts = inputs["text"]
            real_imgs = inputs["image"].to(self.args.device)

            # G forward
            text_embeds_g = self.model.text_encoder(batch_texts)
            fake_imgs_g = self.model.generator(text_embeds_g)

            # D 평가
            with torch.no_grad():
                d_real = self.model.discriminator(real_imgs).mean()
                d_fake = self.model.discriminator(fake_imgs_g).mean()

            margin = 0.1
            if d_fake < d_real - margin:
                loss_g = -torch.mean(self.model.discriminator(fake_imgs_g))

                self.opt_g.zero_grad()
                loss_g.backward()
                self.opt_g.step()
            else:
                # G 학습 스킵
                pass

        return total_loss.detach()



def custom_data_collator(features):
    texts = [f["text"] for f in features]
    images = torch.stack([f["image"] for f in features])
    return {
        "text": texts,
        "image": images
    }

def visualize_real_dataset_samples(dataset, num_samples=5):
    fig, axes = plt.subplots(1, num_samples, figsize=(4*num_samples, 4))

    for i in range(num_samples):
        sample = dataset[i]
        img = sample["image"]
        caption = sample["text"]

        np_img = img.permute(1, 2, 0).numpy()

        axes[i].imshow(np_img)
        axes[i].set_title(f"{caption[:40]}...", fontsize=10)
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":

    torch.autograd.set_detect_anomaly(True)
    
    token = "" 
    dataset = load_dataset("laion/laion400m", split="train[:1000]", token=token)
    dset = LAIONDataset(dataset)
    
    training_args = TrainingArguments(
        output_dir="./gan_output",
        per_device_train_batch_size=4,
        num_train_epochs=50,
        logging_steps=10,
        save_strategy="epoch",
        remove_unused_columns=False,
        report_to="none"
    )
    
    text_encoder = TextEncoder().to("cuda")
    generator = Generator().to("cuda")
    discriminator = Discriminator().to("cuda")

    gan_model = GANModule(generator, discriminator, text_encoder)

    trainer = GANTrainer(
        model=gan_model,
        args=training_args,
        train_dataset=dset,
        data_collator=custom_data_collator,
        callbacks=[GANSaveCallback(save_dir="trained_gan_generator", sample_dir="gan_samples", device="cuda")]
    )

    trainer.train()