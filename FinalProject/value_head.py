import torch.nn as nn
import torch
import torch.nn.functional as F
from transformers import RobertaModel, RobertaTokenizer

class ValueHead(nn.Module):
    def __init__(self, hidden_dim, intermediate_dim=512):
        super().__init__()
        self.encoder = RobertaModel.from_pretrained("microsoft/codebert-base")
        self.tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")

        hidden_dim = self.encoder.config.hidden_size

        # 각 용도에 따라 네 개의 분리된 헤드
        self.heads = nn.ModuleDict({
            "h_disc": nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, intermediate_dim),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(intermediate_dim, 1)
            ),
            "h_format": nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, intermediate_dim),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(intermediate_dim, 1)
            ),
            "cpp_disc": nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, intermediate_dim),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(intermediate_dim, 1)
            ),
            "cpp_format": nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, intermediate_dim),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(intermediate_dim, 1)
            )
        })

    def forward(self, input_ids, attention_mask, mode: str = "h"):
        assert mode in ["h", "cpp"], f"Invalid mode: {mode}"

        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_rep = outputs.last_hidden_state[:, 0, :]

        v_disc = self.heads[f"{mode}_disc"](cls_rep)
        v_format = self.heads[f"{mode}_format"](cls_rep)
        v_total = v_disc + v_format

        return v_total, v_disc, v_format

