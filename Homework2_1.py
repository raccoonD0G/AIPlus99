from torch.utils.data import DataLoader
from diffusers import StableDiffusionPipeline
from torchvision import transforms
import kagglehub
import random
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Adam
import matplotlib.pyplot as plt
import hashlib
import os
from PIL import Image
from io import BytesIO
from transformers import DistilBertTokenizer, DistilBertModel
from image_autoencoder import ImageEncoder
from tqdm import tqdm

def load_data(path, nrows=None):
    df = pd.read_csv(path, nrows=nrows, keep_default_na=False)
    data = []
    for _, row in df.iterrows():
        if len(row['premise']) * len(row['hypothesis']) != 0:
            data.append({'premise': row['premise'], 'hypothesis': row['hypothesis'], 'label': row['label']})

    return data

# 데이터를 배치로 묶기 위한 함수 정의
def collate_fn(batch):
    premises, hypothesises, labels = [], [], []

    for row in batch:
        premises.append(row['premise'])
        hypothesises.append(row['hypothesis'])
        labels.append(row['label'])

    sentence_pairs = list(zip(premises, hypothesises))

    labels = torch.tensor(labels)

    return sentence_pairs, labels

class CrossAttention(nn.Module):
    def __init__(self, dim=768, heads=8, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, batch_first=True)
        self.ln1 = nn.LayerNorm(dim)

        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )
        self.ln2 = nn.LayerNorm(dim)

    def forward(self, query_feat, context_feat):
        # Cross-Attention
        attn_out, _ = self.attn(query=query_feat, key=context_feat, value=context_feat)
        x = self.ln1(attn_out + query_feat)

        # Feed-Forward Network
        ffn_out = self.ffn(x)
        x = self.ln2(x + ffn_out)
        return x

class TextToImageClassifier(nn.Module):
    def __init__(self, premises=[], hypotheses=[]):
        super().__init__()

        self.image_cache = {}

        if premises and hypotheses:
            self.generate_images_if_needed(premises, hypotheses)
            self.preload_images(premises, hypotheses)
            tqdm.write(f"Preloaded {len(self.image_cache)} images into memory.")

        # BERT tokenizer & 모델
        self.tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
        self.bert = DistilBertModel.from_pretrained("distilbert-base-uncased")

        for name, param in self.bert.named_parameters():
            if "transformer.layer.4" in name or "transformer.layer.5" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        # Stable Diffusion 모델
        self.pipe = StableDiffusionPipeline.from_pretrained(
            "segmind/tiny-sd",
            torch_dtype=torch.float16,
            safety_checker=None,
            feature_extractor=None
        ).to("cuda")

        self.pipe.enable_attention_slicing()

        # StableDiffusion 을 완전히 얼림, 학습시키지 않을 예정
        for param in self.pipe.unet.parameters():
            param.requires_grad = False

        for param in self.pipe.vae.parameters():
            param.requires_grad = False

        for param in self.pipe.text_encoder.parameters():
            param.requires_grad = False

        # 이미지 전처리 (train용, test용)
        self.train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomChoice([
                transforms.RandomRotation(degrees=10),
                transforms.RandomHorizontalFlip(p=1.0),
                transforms.Lambda(lambda x: x)
            ]),
            transforms.ToTensor(),
        ])

        self.eval_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

        self.image_encoder = ImageEncoder().to("cuda")

        self.ln_image = nn.LayerNorm(768)
        self.ln_text = nn.LayerNorm(768)

        self.cross_attn_ti = CrossAttention()  # Text ← Image
        self.cross_attn_it = CrossAttention()  # Image ← Text

        # 결합 후 분류기 (이미지 + 텍스트)
        self.classifier = nn.Sequential(
            nn.Linear(768 + 768, 512),
            nn.LayerNorm(512),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),

            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),

            nn.Linear(128, 3)
        )

        # 가중치 로드 추가
        save_path = "text_classifier.pt"
        if os.path.exists(save_path):
            self.load_state_dict(torch.load(save_path, map_location="cuda"))
            tqdm.write(f"가중치 로드 완료: {save_path}")
        else:
            tqdm.write(f"저장된 가중치가 없어 처음부터 학습합니다.")

    # 이미지 파일 경로 반환
    def prompt_to_image_filename(self, premise, hypothesis):
        hash_key = hashlib.md5((premise + hypothesis).encode()).hexdigest()
        return os.path.join("image_save", f"{hash_key}.png")

    def load_image(self, path):
        with open(path, "rb") as f:
            img_bytes = f.read()
        img = Image.open(BytesIO(img_bytes))
        return img.convert("RGB") if img.mode != "RGB" else img

    def load_image_cached(self, path):
        if path in self.image_cache:
            return self.image_cache[path]
        img = self.load_image(path)
        self.image_cache[path] = img
        return img

    def preload_images(self, premises, hypotheses):
        for prem, hypo in zip(premises, hypotheses):
            path = self.prompt_to_image_filename(prem, hypo)
            if os.path.exists(path):
                self.image_cache[path] = self.load_image(path)

    def generate_images_if_needed(self, premises, hypotheses):
        need_generate = []
        for prem, hypo in zip(premises, hypotheses):
            path = self.prompt_to_image_filename(prem, hypo)
            if not os.path.exists(path):
                need_generate.append((prem, hypo))

        if not need_generate:
            return

        p_prompts = [p for p, _ in need_generate]
        h_prompts = [h for _, h in need_generate]

        with torch.no_grad():
            p_imgs = self.pipe(p_prompts, num_inference_steps=10, progress_bar=True).images
            h_imgs = self.pipe(h_prompts, num_inference_steps=10, progress_bar=True).images

        for (prem, hypo), p_img, h_img in zip(need_generate, p_imgs, h_imgs):
            combined = Image.new("RGB", (p_img.width + h_img.width, p_img.height))
            combined.paste(p_img, (0, 0))
            combined.paste(h_img, (p_img.width, 0))
            combined.save(self.prompt_to_image_filename(prem, hypo))

    def forward(self, sentence_pairs):
        images, texts = [], []

        # 문장 분리
        premises = [p for p, _ in sentence_pairs]
        hypotheses = [h for _, h in sentence_pairs]

        # 1. 이미지 경로 생성 및 이미지 필요 시 생성
        img_paths = [self.prompt_to_image_filename(p, h) for p, h in sentence_pairs]
        self.generate_images_if_needed(premises, hypotheses)

        # 2. 이미지 로드 및 텍스트 인코딩
        for (premise, hypothesis), img_path in zip(sentence_pairs, img_paths):
            combined_text = f"{premise} [SEP] {hypothesis}"
            combined = self.load_image_cached(img_path)

            if combined.mode != "RGB":
                combined = combined.convert("RGB")

            w = combined.width // 2
            p_img = combined.crop((0, 0, w, combined.height))
            h_img = combined.crop((w, 0, combined.width, combined.height))

            combo = Image.new("RGB", (p_img.width + h_img.width, p_img.height))
            combo.paste(p_img, (0, 0))
            combo.paste(h_img, (p_img.width, 0))

            transform = self.train_transform if self.training else self.eval_transform
            images.append(transform(combo))
            texts.append(combined_text)

        # 3. Encoder + BERT + Classifier
        images = torch.stack(images).to("cuda")  # (B, 3, 224, 224)
        img_features = self.image_encoder(images)

        encoded = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to("cuda")
        bert_outputs = self.bert(**encoded).last_hidden_state[:, 0, :]  # (B, 768)

        img_features = self.ln_image(img_features)
        bert_outputs = self.ln_text(bert_outputs)

        # BERT + ImageEncoder 결과
        text_feat = bert_outputs.unsqueeze(1)  # (B, 1, 768)
        image_feat = img_features.unsqueeze(1)  # (B, 1, 768)

        # Cross Attention 각각 적용
        fused_text = self.cross_attn_ti(text_feat, image_feat)  # Text <- Image
        fused_img = self.cross_attn_it(image_feat, text_feat)  # Image <- Text

        # flatten 후 concat
        fused = torch.cat([fused_text.squeeze(1), fused_img.squeeze(1)], dim=-1)  # (B, 1536)

        return self.classifier(fused)

    def extract_features(self, sentence_pairs):
        images, texts = [], []

        # 문장 분리
        premises = [p for p, _ in sentence_pairs]
        hypotheses = [h for _, h in sentence_pairs]

        # 이미지 경로 및 생성
        img_paths = [self.prompt_to_image_filename(p, h) for p, h in sentence_pairs]
        self.generate_images_if_needed(premises, hypotheses)

        for (premise, hypothesis), img_path in zip(sentence_pairs, img_paths):
            combined_text = f"{premise} [SEP] {hypothesis}"
            combined = self.load_image_cached(img_path)

            if combined.mode != "RGB":
                combined = combined.convert("RGB")

            w = combined.width // 2
            p_img = combined.crop((0, 0, w, combined.height))
            h_img = combined.crop((w, 0, combined.width, combined.height))

            combo = Image.new("RGB", (p_img.width + h_img.width, p_img.height))
            combo.paste(p_img, (0, 0))
            combo.paste(h_img, (p_img.width, 0))

            transform = self.train_transform if self.training else self.eval_transform
            images.append(transform(combo))
            texts.append(combined_text)

        # 인코딩
        images = torch.stack(images).to("cuda")
        img_features = self.image_encoder(images)

        encoded = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to("cuda")
        bert_outputs = self.bert(**encoded).last_hidden_state[:, 0, :]

        img_features = self.ln_image(img_features)
        bert_outputs = self.ln_text(bert_outputs)

        text_feat = bert_outputs.unsqueeze(1)
        image_feat = img_features.unsqueeze(1)

        fused_text = self.cross_attn_ti(text_feat, image_feat)
        fused_img = self.cross_attn_it(image_feat, text_feat)

        fused = torch.cat([fused_text.squeeze(1), fused_img.squeeze(1)], dim=-1)  # (B, 1536)
        return fused


def accuracy(model, dataloader):
    cnt = 0
    acc = 0

    model.eval()

    for data in dataloader:
        sentence_pairs, labels = data
        labels = labels.to('cuda')

        preds = model(sentence_pairs)
        preds = torch.argmax(preds, dim=-1)

        cnt += labels.size(0)
        acc += (preds == labels).sum().item()

    return acc / cnt

if __name__ == "__main__":
    path = kagglehub.dataset_download("thedevastator/unlocking-language-understanding-with-the-multin")
    tqdm.write("Path to dataset files:" + path)

    # 데이터가 부족한 상황을 가정하여 1000개만 로드
    train_ds = load_data(path + '/train.csv', nrows=1000)
    test_ds = load_data(path + '/validation_matched.csv', nrows=1000)

    # 학습용 DataLoader 정의 (shuffle=True로 배치 순서 랜덤화)
    train_loader = DataLoader(
        train_ds, batch_size=32, shuffle=True, collate_fn=collate_fn
    )

    # 테스트용 DataLoader 정의 (shuffle=False로 배치 순서 고정)
    test_loader = DataLoader(
        test_ds, batch_size=32, shuffle=False, collate_fn=collate_fn
    )

    model = TextToImageClassifier().to("cuda")

    # 학습 설정
    lr = 5e-5
    # 과한 확신을 막기 위한 Label Smoothing
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

    optimizer = Adam(model.parameters(), lr=lr)
    n_epochs = 100

    # 정확도 및 손실 기록용 리스트
    train_losses = []
    train_accuracies = []
    test_accuracies = []

    # 학습 루프
    for epoch in range(n_epochs):
        total_loss = 0.
        model.train()

        for data in tqdm(train_loader, desc=f"Epoch {epoch+1}/{n_epochs}"):
            model.zero_grad()

            sentence_pairs, labels = data
            labels = labels.to('cuda').long()

            preds = model(sentence_pairs)
            loss = loss_fn(preds, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # 손실 기록
        train_losses.append(total_loss)
        tqdm.write(f"Epoch {epoch:3d} | Train Loss: {total_loss:.4f}")

        save_path = "text_classifier.pt"
        torch.save(model.state_dict(), save_path)
        tqdm.write(f"모델 가중치 저장 완료: {save_path}")

        # 평가
        with torch.no_grad():
            model.eval()
            train_acc = accuracy(model, train_loader)
            test_acc = accuracy(model, test_loader)

            tqdm.write(f"=========> Train acc: {train_acc:.3f} | Test acc: {test_acc:.3f}")

            train_accuracies.append(train_acc)
            test_accuracies.append(test_acc)

            if epoch % 5 == 0:
                tqdm.write("\n샘플 예측 결과 (무작위 5개):\n")
                model.eval()

                random_samples = random.sample(test_ds, 5)  # 20개 무작위 샘플 추출
                sentence_pairs, labels = collate_fn(random_samples)
                labels = labels.to('cuda')
                preds = model(sentence_pairs)
                preds = torch.argmax(preds, dim=-1)

                for i in range(len(sentence_pairs)):
                    prem, hypo = sentence_pairs[i]
                    true_label = labels[i].item()
                    pred_label = preds[i].item()

                    tqdm.write(f"Premise   : {prem}")
                    tqdm.write(f"Hypothesis: {hypo}")
                    tqdm.write(f"Label: {true_label} | Predicted: {pred_label}")
                    tqdm.write("-" * 80)


    plt.figure(figsize=(14, 5))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Train Loss")
    plt.title("Train Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid()
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label="Train Acc")
    plt.plot(test_accuracies, label="Test Acc")
    plt.title("Accuracy Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.grid()
    plt.legend()

    plt.tight_layout()
    plt.show()