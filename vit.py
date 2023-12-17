import numpy as np
import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    '''
    Flatten the patches and map to D dimensions with a trainable linear projection.
    '''
    def __init__(self, config_dict):
        super().__init__()
        self.img_size = config_dict['img_size']
        self.img_channels = config_dict['img_channels']
        self.patch_size = config_dict['patch_size']
        self.hidden_size = config_dict['hidden_size']

        self.projection = nn.Conv2d(
            in_channels = self.img_channels,
            out_channels = self.hidden_size,
            stride = self.patch_size,
            kernel_size = self.patch_size,
        )

    def forward(self, x):
        x = self.projection(x)
        x = x.flatten(2).transpose(1, 2)
        return x
    
class TokenEmbedding(nn.Module):
    '''
    Similar to BERTs [class] token, we prepend a learnable embedding to the sequence of embedded patches.
    '''
    def __init__(self, config_dict):
        super().__init__()
        self.hidden_size = config_dict['hidden_size']

        self.token = nn.Parameter(torch.randn(1, 1, self.hidden_size))

    def forward(self, x):
        batch_size = x.size()[0]
        tokens = self.token.expand(batch_size, -1, -1)
        x = torch.cat((tokens, x), dim=1)
        return x

class PositionalEmbedding(nn.Module):
    '''
    Position embeddings are added to the patch embeddings to retain positional information.
    '''
    def __init__(self, config_dict):
        super().__init__()
        self.hidden_size = config_dict['hidden_size']
        self.num_patches = (config_dict['img_size'] // config_dict['patch_size']) ** 2
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches+1, self.hidden_size))

    def forward(self, x):
        x = x + self.pos_embedding
        return x
    
class AttentionLayer(nn.Module):
    '''
    Compute Scale Dot-Product Attention
    '''
    def __init__(self, config_dict):
        super().__init__()
        self.hidden_size = config_dict['hidden_size']
        self.attention_out_size = self.hidden_size // config_dict['num_attention_heads']

        self.Q = nn.Linear(self.hidden_size, self.attention_out_size)
        self.K = nn.Linear(self.hidden_size, self.attention_out_size)
        self.V = nn.Linear(self.hidden_size, self.attention_out_size)

    def forward(self, x):
        Q = self.Q(x)
        K = self.K(x)
        V = self.V(x)

        attention = torch.matmul(Q, K.transpose(-1, -2))
        attention = attention / np.sqrt(self.attention_out_size)
        attention = nn.functional.softmax(attention, dim=-1)
        attention = torch.matmul(attention, V)

        return attention
    
class MultiHeadAttention(nn.Module):
    '''
    Concat Multiple Attention Layers and pass output to Linear
    '''
    def __init__(self, config_dict):
        super().__init__()
        self.hidden_size = config_dict['hidden_size']
        self.num_attention_heads = config_dict['num_attention_heads']
        self.attention_head_size = self.hidden_size // self.num_attention_heads
        self.mh_output_size = self.num_attention_heads * self.attention_head_size

        self.heads = nn.ModuleList([])
        self.output_linear = nn.Linear(self.mh_output_size, self.hidden_size)

        for _ in range(self.num_attention_heads):
            head = AttentionLayer(config_dict)
            self.heads.append(head)

    def forward(self, x):
        mh_output = [head(x) for head in self.heads]
        mh_output = torch.cat([output for output in mh_output], dim=-1)
        mh_output = self.output_linear(mh_output)

        return mh_output
    
class MLP(nn.Module):
    '''
    The MLP contains two layers with a GELU non-linearity.
    '''
    def __init__(self, config_dict):
        super().__init__()
        self.hidden_size = config_dict['hidden_size']
        self.mlp_hidden_size = config_dict['mlp_hidden_size']

        self.layer_1 = nn.Linear(self.hidden_size,self.mlp_hidden_size)
        self.layer_2 = nn.Linear(self.mlp_hidden_size,self.hidden_size)

        self.activation_fcn = nn.GELU()

    def forward(self, x):
        x = self.layer_1(x)
        x = self.activation_fcn(x)
        x = self.layer_2(x)

        return x
    
class EncoderBlock(nn.Module):
    '''
    The MLP contains two layers with a GELU non-linearity.
    '''
    def __init__(self, config_dict):
        super().__init__()
        self.mh_attention = MultiHeadAttention(config_dict)
        self.norm_1 = nn.LayerNorm(config_dict['hidden_size'])
        self.mlp = MLP(config_dict)
        self.norm_2 = nn.LayerNorm(config_dict['hidden_size'])

    def forward(self, x):
        attention_out = self.norm_1(x)
        attention_out = self.mh_attention(attention_out)
        x = attention_out + x #skip connection
        mlp_out = self.norm_2(x)
        mlp_out = self.mlp(x)
        x = mlp_out + x #skip connection

        return x
    
class TransformerEncoder(nn.Module):
    '''
    Transformer Encoder Structure.
    '''
    def __init__(self, config_dict):
        super().__init__()
        self.blocks = nn.ModuleList([])
        for _ in range(config_dict['num_hidden_layers']):
            block = EncoderBlock(config_dict)
            self.blocks.append(block)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)

        return x
    
class ViT(nn.Module):

    def __init__(self, config_dict):
        super().__init__()
        self.config = config_dict
        self.patch_embedding = PatchEmbedding(config_dict)
        self.token_embedding = TokenEmbedding(config_dict)
        self.pos_embedding = PositionalEmbedding(config_dict)
        self.encoder = TransformerEncoder(config_dict)
        self.hidden_layer = nn.Linear(config_dict['hidden_size'],config_dict['hidden_size'])
        self.tanh = nn.Tanh()
        self.cls_layer = nn.Linear(config_dict['hidden_size'],config_dict['num_classes'])
        self.transfer_learning = config_dict['transfer_learning']

        self.apply(self._init_weights)

    def forward(self, x):
        if self.transfer_learning:
            with torch.no_grad():
                # Embedding
                x = self.patch_embedding(x)
                x = self.token_embedding(x)
                x = self.pos_embedding(x)

                # Transformer Encoder
                x = self.encoder(x)
        else:
            # Embedding
            x = self.patch_embedding(x)
            x = self.token_embedding(x)
            x = self.pos_embedding(x)

            # Transformer Encoder
            x = self.encoder(x)

        # Classifier
        x = self.hidden_layer(x[:,0,:])
        x = self.tanh(x)
        x = self.cls_layer(x)

        return x
    
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            torch.nn.init.normal_(module.weight,
                mean=0.0,
                std=self.config["initializer_range"])
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, PositionalEmbedding):
            module.pos_embedding.data = nn.init.trunc_normal_(
                module.pos_embedding.data.to(torch.float32),
                mean=0.0,
                std=self.config["initializer_range"],
            ).to(module.pos_embedding.dtype)
        elif isinstance(module, TokenEmbedding):
            module.token.data = nn.init.trunc_normal_(
                module.token.data.to(torch.float32),
                mean=0.0,
                std=self.config["initializer_range"],
            ).to(module.token.dtype)