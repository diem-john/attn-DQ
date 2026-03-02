import transformers as ts
import qtransformers as qts
import torch.nn as nn


class TransformerModel(nn.Module):
    def __init__(self, input_dim=5, d_model=128, nhead=4, num_layers=4, target_window=36, dropout=0.4):
        super(TransformerModel, self).__init__()

        self.encoder = ts.Linear(input_dim, d_model)
        self.pos_encoder = ts.PositionalEncoding(d_model, dropout)
        encoder_layers = ts.TransformerEncoderLayer(d_model, nhead)
        self.transformer_encoder = ts.TransformerEncoder(encoder_layers, num_layers)
        # self.decoder = ts.Linear(d_model, target_window)
        self.pre_net = ts.Linear(d_model, 128)
        self.decoder = ts.Linear(128, d_model)
        self.post_net = ts.Linear(d_model, target_window)

    def forward(self, x):
        x = self.encoder(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = self.pre_net(x)
        x = self.decoder(x[:, -1, :])
        x = self.post_net(x)
        return x


class QTransformerModel(nn.Module):
    def __init__(self, input_dim=5, d_model=128, nhead=4, num_layers=4, target_window=36, dropout=0.4):
        super(QTransformerModel, self).__init__()

        self.encoder = ts.Linear(input_dim, d_model)
        self.pos_encoder = ts.PositionalEncoding(d_model, dropout)
        encoder_layers = qts.TransformerEncoderLayer(d_model, nhead)
        self.transformer_encoder = qts.TransformerEncoder(encoder_layers, num_layers)
        self.pre_net = ts.Linear(d_model, 128)
        self.decoder = ts.Linear(128, 4)
        self.post_net = ts.Linear(4, target_window)

    def forward(self, x):
        x = self.encoder(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = self.pre_net(x)
        x = self.decoder(x[:, -1, :])
        x = self.post_net(x)
        return x

class QTransformerModel_ED(nn.Module):
    def __init__(self, input_dim=5, d_model=128, nhead=4, num_layers=4, target_window=36, dropout=0.4):
        super(QTransformerModel_ED, self).__init__()

        self.encoder = ts.Linear(input_dim, d_model)
        self.pos_encoder = ts.PositionalEncoding(d_model, dropout)
        encoder_layers = qts.TransformerEncoderLayer(d_model, nhead)
        self.transformer_encoder = qts.TransformerEncoder(encoder_layers, num_layers)
        self.pre_net = ts.Linear(d_model, 128)
        self.decoder = qts.QuantumDress()
        self.post_net = ts.Linear(4, target_window)

    def forward(self, x):
        x = self.encoder(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = self.pre_net(x)
        x = self.decoder(x[:, -1, :])
        x = self.post_net(x)
        return x

class QTransformerModel_D(nn.Module):
    def __init__(self, input_dim=5, d_model=128, nhead=4, num_layers=4, target_window=36, dropout=0.4):
        super(QTransformerModel_ED, self).__init__()

        self.encoder = ts.Linear(input_dim, d_model)
        self.pos_encoder = ts.PositionalEncoding(d_model, dropout)
        encoder_layers = ts.TransformerEncoderLayer(d_model, nhead)
        self.transformer_encoder = qts.TransformerEncoder(encoder_layers, num_layers)
        self.pre_net = ts.Linear(d_model, 128)
        self.decoder = qts.QuantumDress()
        self.post_net = ts.Linear(4, target_window)

    def forward(self, x):
        x = self.encoder(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = self.pre_net(x)
        x = self.decoder(x[:, -1, :])
        x = self.post_net(x)
        return x
