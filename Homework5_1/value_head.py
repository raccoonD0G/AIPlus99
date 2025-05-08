import torch.nn as nn

class ValueHead(nn.Module):
    def __init__(self, hidden_dim, n_layers=2, n_heads=8):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            batch_first=True,
            activation='gelu'
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.value_disc = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.GELU(),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, 1)
        )
        self.value_format = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.GELU(),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, 1)
        )

    def forward(self, hidden_states, attention_mask=None):
 
        if attention_mask is not None:
            src_key_padding_mask = ~attention_mask.bool()
        else:
            src_key_padding_mask = None

        encoded = self.encoder(hidden_states, src_key_padding_mask=src_key_padding_mask)

        cls_rep = encoded[:, -1, :]

        v_disc = self.value_disc(cls_rep)
        v_format = self.value_format(cls_rep)
        v_total = v_disc + v_format

        return v_total, v_disc, v_format
