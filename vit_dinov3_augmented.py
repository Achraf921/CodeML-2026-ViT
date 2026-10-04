import torch
import torch.nn.functional as F
import torch.nn as nn
from rotary_embedding_torch import RotaryEmbedding, apply_rotary_emb # pytorch somehow doesn't have a rope implementation 
# (which surprised me) and since I did not feel like making sure that I was not sparking a big ass rotation matrix full of zeros in
# my VRAM I'd rather use this library
from transformers import AutoModel

# ----- Vision Transformer : Pixels -> features ------


#----- Here we have used the DINOV3 foundation vision model from Meta Fair instead of our Cnn inn the ConvLayer setup due to our training data being so tiny
# praying that we get some transfer learning benefits out of it (we also fine-tune it as its gradients are included in the gradient tree)

class ViT_model(nn.Module): 
    def __init__(self,B, H, W, n_head, ff_layers, ff_embd, n_blocks, div_H, div_W,n_embd): # C is 3 as r,g,b
        super().__init__()
        assert H%div_H == 0 and W%div_W==0
        self.B = B
        self.H = H
        self.W = W
        self.n_embd = n_embd
        self.div_H = div_H # 128
        self.div_W = div_W # 128
        self.conv_layer = ConvLayer(n_embd) # runs the stem cnn and tokenizes down *1/128 on both H and W, upscaling the rgb to n_embd
        self.old_h = H//div_H
        self.old_w = W//div_W
        self.blocks = nn.ModuleList(Block(B, H*W//(div_H*div_W), n_embd, n_head, ff_layers, ff_embd, self.old_h, self.old_w) for _ in range(n_blocks))
        self.output = OutputScaleLayer() # should get us to the output dimension

    def forward(self, x):
        out = self.conv_layer(x)
        B, C, h, w = out.shape
        # we'll now need to work in a H,W dimspace, hence we'll concat our columns through tensor ops to get to a (H.W, C) rather than (C, H, W)
        out = out.permute(0, 2, 3, 1).reshape(B, h*w, C) # we swap back C to the third dim and then reshape such that H.W feeds from W
        for b in self.blocks:
            out = b(out)
        out = self.output(out)
        return out

# Blocks
class Block(nn.Module):
    def __init__(self, B, H, W, n_head, ff_layers, ff_embd, old_h, old_w):
        super().__init__()
        # input arrives in (H*W, C) format (that come in as H and W)
        self.ln_1 = nn.LayerNorm(W)
        self.attention = Attention(B, H, W, n_head, old_h, old_w)
        self.mlp = MLP(H, W, ff_layers, ff_embd)
        self.l2 = nn.LayerNorm(W)
    
    def forward(self, x): # x arrives in H*W, C as mentioned
        out = self.ln_1(x)
        x2 = x + self.attention(out)
        out = self.l2(x2) #2nd layer norm
        out = x2 + self.mlp(out)
        return out


# Attention 

class Attention(nn.Module):
    def __init__(self,B , H, W, n_head, old_h, old_w):
        super().__init__()
        self.H = H
        self.W = W
        self.n_head = n_head #
        assert W % n_head == 0
        self.attention_heads = nn.ModuleList([AttentionHead(B, H, W, (W//n_head), old_h, old_w) for _ in range(n_head)])
        self.mix = nn.Linear(W,W) # mixing layer

    def forward(self,x):
        # x is in H*W, C format as you remember

        out = x
        outArray = []

        for head in self.attention_heads:
            outArray = outArray + [head(out)]
        
        out = torch.cat(outArray, dim=-1)# concatenating back
        out = self.mix(out) # mixing the embeddings back
        return out 

# Attention heads
class AttentionHead(nn.Module):
    def __init__(self, B, H, og_W, W, old_h, old_w): # we also had to cascade down fucking old_h and old_w due to the rope's library format requiring them
        super().__init__()
        #matrices to create our QKV stuff
        self.B=B
        self.H = H
        self.W = W
        self.old_h = old_h
        self.old_w = old_w
        self.Wq = nn.Linear(og_W,W)
        self.Wk = nn.Linear(og_W,W)
        self.Wv = nn.Linear(og_W,W)
        self.rope = RotaryEmbedding(dim= W // 2, freqs_for='pixel', max_freq=256)
    
    def forward(self, x):
        K = self.Wk(x)
        Q = self.Wq(x)
        V = self.Wv(x)

        # Here we will use RoPe positional embeddings, rather than just a linear layer
        # (Rotary Positional embeddings)
        # to get both absolute and relative positional embeddings
        # this preserves the "angle" between token embedding vectors while also providing absolute embedding data
        # (through applying our rotation matrix * m.theta, m depending on its position, each theta specific to a pair of dimensions in the embds)

        freqs = self.rope.get_axial_freqs(self.old_h, self.old_w)
        K = apply_rotary_emb(freqs, K.view(self.B, self.old_h, self.old_w, -1)).reshape(self.B, self.H, -1)
        Q = apply_rotary_emb(freqs, Q.view(self.B, self.old_h, self.old_w, -1)).reshape(self.B, self.H, -1) # we need to play around with the shape as the library expects this fuck ass shape
        # applying all of our relative rotations to Q and K pre attention formula but not V as specified in the paper

        out = F.scaled_dot_product_attention(Q,K,V, attn_mask=None) # Fast attenton, + no lower trig matrix mask
        return out

#MLP
class MLP(nn.Module):
    def __init__(self, H, W, ff_layer, ff_embd):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(W,ff_embd)] + [nn.Linear(ff_embd,ff_embd) for _ in range(ff_layer-1)])
        self.output_layer = nn.Linear(ff_embd, W)

    def forward(self, x):
        out = x
        for layer in self.layers:
            out = layer(out)
            out = F.gelu(out) # activation function
        out = self.output_layer(out)
        return out

#Initial conv layer

class ConvLayer(nn.Module):
    def __init__(self, n_embd, name="facebook/dinov3-convnext-tiny-pretrain-lvd1689m"): # 44M parameters
        super().__init__()
        self.backbone = AutoModel.from_pretrained(name)              # pretrained, requires_grad=True by default
        c = self.backbone.config.hidden_sizes[-1]                     # 768 for Tiny
        self.proj = nn.Conv2d(c, n_embd, kernel_size=4, stride=4)     # /32 -> /128, 768 -> n_embd
        # DINOv3 expects ImageNet-normalized input
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std",  torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):                                             # (B, 3, H, W), floats in [0, 1]
        x = (x - self.mean) / self.std
        feat = self.backbone(pixel_values=x, output_hidden_states=True).hidden_states[-1]   # (B, 768, H/32, W/32)
        return self.proj(feat)

# final deconv back to the output format 

class OutputScaleLayer(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return x
        