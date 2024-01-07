import torch
import torch.nn as nn


class InputEmbedding(nn.Module):
    def __init__(self, args, seq_length, input_size):
        super().__init__()
        self.embed_dim = args.embed_dim
        self.linear = nn.Linear(in_features=input_size, out_features=self.embed_dim)
        self.class_token = nn.Parameter(torch.randn(1, 1, self.embed_dim))
        self.pos_encoding = nn.Parameter(torch.randn(1, seq_length + 1, self.embed_dim))

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.linear(x)
        class_token = torch.broadcast_to(self.class_token, (batch_size, 1, self.embed_dim))
        x = torch.cat([class_token, x], dim=1)
        x += self.pos_encoding
        return x


class KWSTransformer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.input_embeddings = self.construct_input_embeddings(args)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=args.embed_dim,
            nhead=args.num_heads,
            dim_feedforward=args.ff_dim,
            dropout=args.dropout,
            activation=nn.GELU(approximate='tanh' if args.approximate_gelu else 'none'),
            batch_first=True,
            norm_first=args.prenorm
        )
        num_inputs = len(self.input_embeddings)
        self.encoders = nn.ModuleList(
            [nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=args.num_layers)] * num_inputs)
        self.linear = nn.Linear(in_features=args.embed_dim * num_inputs, out_features=args.label_count)

    def construct_input_embeddings(self, args):
        if args.attn_type == 'time':
            return nn.ModuleList([InputEmbedding(args, args.num_time_steps, args.num_freq_bins)])
        if args.attn_type == 'freq':
            return nn.ModuleList([InputEmbedding(args, args.num_freq_bins, args.num_time_steps)])
        if args.attn_type == 'both':
            return nn.ModuleList([
                InputEmbedding(args, args.num_time_steps, args.num_freq_bins),
                InputEmbedding(args, args.num_freq_bins, args.num_time_steps),
            ])
        if args.attn_type == 'patch':
            return nn.ModuleList([InputEmbedding(args, args.num_patches, args.patch_size)])

    def forward(self, x):
        x = [self.encoders[index](self.input_embeddings[index](x_i))[:, 0] for index, x_i in enumerate(x)]
        x = torch.cat(x, dim=-1)
        x = self.linear(x)
        return x
