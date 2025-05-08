import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

class CodeDiscriminator(nn.Module):
    def __init__(
        self,
        model_name="microsoft/codebert-base",
        max_length=256,
        hidden_dim=512
    ):
        super().__init__()

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.encoder = AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            trust_remote_code=True
        )

        self.classifier = nn.Sequential(
            nn.LayerNorm(self.encoder.config.hidden_size),
            nn.Linear(self.encoder.config.hidden_size, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1)
        )

        self.max_length = max_length

    def forward(self, code_list):
        inputs = self.tokenizer(
            code_list,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=self.max_length
        ).to(self.encoder.device)

        outputs = self.encoder(**inputs)
        pooled = outputs.last_hidden_state[:, 0, :]

        logits = self.classifier(pooled).squeeze(-1)
        logits = logits.clamp(min=-10, max=10)
        return logits
