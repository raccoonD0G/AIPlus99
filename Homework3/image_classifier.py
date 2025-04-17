import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from datasets import load_dataset
from transformers import (
    DistilBertTokenizer,
    DistilBertModel,
    Trainer,
    TrainingArguments,
    TrainerCallback
)
import matplotlib.pyplot as plt
from image_autoencoder import ImageEncoder
from image_gan import Generator

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

class LoggingCallback(TrainerCallback):
    def __init__(self):
        self.train_losses = []
        self.eval_accuracies = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        if "loss" in logs:
            self.train_losses.append(logs["loss"])
        if "eval_accuracy" in logs:
            self.eval_accuracies.append(logs["eval_accuracy"])

    def plot(self):
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label="Train Loss")
        plt.title("Train Loss over Epochs")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.grid(True)
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(self.eval_accuracies, label="Eval Accuracy", color="green")
        plt.title("Validation Accuracy over Epochs")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.grid(True)
        plt.legend()

        plt.tight_layout()
        plt.show()

class CrossAttention(nn.Module):
    def __init__(self, dim=768, heads=8, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, batch_first=True)
        
        self.ln1_pre = nn.LayerNorm(dim)
        self.ln1_post = nn.LayerNorm(dim)
        
        self.ln2_pre = nn.LayerNorm(dim)
        self.ln2_post = nn.LayerNorm(dim)

        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )

    def forward(self, query_feat, context_feat):
        query_norm = self.ln1_pre(query_feat)
        context_norm = self.ln1_pre(context_feat)

        attn_out, _ = self.attn(query=query_norm, key=context_norm, value=context_norm)
        x = self.ln1_post(attn_out + query_feat)  # Residual + LayerNorm

        ffn_in = self.ln2_pre(x)
        ffn_out = self.ffn(ffn_in)
        out = self.ln2_post(ffn_out + x)  # Residual + LayerNorm

        return out

class SimpleTextToImageClassifier(nn.Module):
    def __init__(self):
        super().__init__()

        self.tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
        self.bert = DistilBertModel.from_pretrained("distilbert-base-uncased")

        for param in self.bert.parameters():
            param.requires_grad = False

        for layer in self.bert.transformer.layer[-2:]:
            for param in layer.parameters():
                param.requires_grad = True

        self.generator = Generator().to("cuda")
        self.image_encoder = ImageEncoder().to("cuda")
        
        image_encoder_weight_path = "trained_image_autoencoder"
        self.image_encoder._try_load_weights(image_encoder_weight_path, "cuda")

        for param in self.image_encoder.parameters():
            param.requires_grad = True

        self.ln_image = nn.LayerNorm(768)
        self.ln_text = nn.LayerNorm(768)

        self.cross_attn_ti = nn.ModuleList([
            CrossAttention(),
            CrossAttention(),
            CrossAttention()
        ])

        self.cross_attn_it = nn.ModuleList([
            CrossAttention(),
            CrossAttention(),
            CrossAttention()
        ])

        self.classifier = nn.Sequential(
            nn.Linear(768 * 2, 512),
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

        self.img_projector = nn.Sequential(
            nn.Linear(1024, 768),
            nn.LayerNorm(768),
            nn.LeakyReLU(0.1),
        )

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :]
        bert_outputs = self.ln_text(bert_outputs)

        fake_imgs = self.generator(bert_outputs)

        img_tensors = nn.functional.interpolate(fake_imgs, size=(224, 224), mode="bilinear", align_corners=False)
        img_features = self.image_encoder(img_tensors)
        img_features = self.img_projector(img_features)
        img_features = self.ln_image(img_features)
        img_features = self.ln_image(img_features)

        text_feat = bert_outputs.unsqueeze(1)
        image_feat = img_features.unsqueeze(1)

        for layer in self.cross_attn_ti:
            text_feat = layer(text_feat, image_feat)

        for layer in self.cross_attn_it:
            image_feat = layer(image_feat, text_feat)

        fused = torch.cat([text_feat.squeeze(1), image_feat.squeeze(1)], dim=-1)
        logits = self.classifier(fused)

        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)
            return loss, logits

        return logits

# MNLI 데이터셋 불러오기
train_data = load_dataset("nyu-mll/glue", "mnli", split="train[:1000]")
val_data = load_dataset("nyu-mll/glue", "mnli", split="validation_matched[:1000]")


# 라벨 맵핑 (0: entailment, 1: neutral, 2: contradiction)
label_map = {"entailment": 0, "neutral": 1, "contradiction": 2}

# 사용자 정의 Dataset 클래스
class MNLIDataset(Dataset):
    def __init__(self, data):
        self.examples = []
        for ex in data:
            text = f"{ex['premise']} [SEP] {ex['hypothesis']}"
            encoding = tokenizer(text, padding="max_length", truncation=True, max_length=128, return_tensors="pt")
            self.examples.append({
                "input_ids": encoding["input_ids"].squeeze(0),
                "attention_mask": encoding["attention_mask"].squeeze(0),
                "labels": torch.tensor(ex["label"])
            })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]

# Trainer에서 사용할 compute_metrics 함수
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = torch.argmax(torch.tensor(logits), dim=-1)
    acc = (preds == torch.tensor(labels)).float().mean().item()
    return {"accuracy": acc}


# 학습 인자
training_args = TrainingArguments(
    output_dir="hf_transformer",
    num_train_epochs=40,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    learning_rate=1e-5,
    logging_strategy="epoch",
    eval_strategy="epoch",
    save_strategy="epoch",
    do_train=True,
    do_eval=True,
    load_best_model_at_end=True,
    save_total_limit=1
)

# 모델 준비
model = SimpleTextToImageClassifier().to("cuda")

# Trainer 정의
logger = LoggingCallback()

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=MNLIDataset(train_data),
    eval_dataset=MNLIDataset(val_data),
    compute_metrics=compute_metrics,
    callbacks=[logger]  # 콜백 등록
)

# 학습 실행
trainer.train()

# 학습 이후 그래프 출력
logger.plot()
