import torch
from vit__dinov3_augmented import ViT_model
import math
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import os, time # we'll use to log to a file each training run

device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.mps.is_available() else "cpu"

# ----- logging files/folders

run = f"runs/vit_{time.strftime('%Y%m%d_%H%M%S')}"
os.makedirs(f"{run}/checkpoints", exist_ok=True)
os.makedirs(f"{run}/samples", exist_ok=True)
log = open(f"{run}/log.csv", "w")


# ------ Let's train that bitch -----

# ----- Hyperparams
B=4
H = 7296
W = 9600 # our image's initial format once passed from pdf to pixels
div_H = 128
div_W = 128 # what our convolutional stem will shrink divide H and W to eventually
n_embd = 384 # our tokens's embedding dim post-patchification
n_head = 6
n_blocks = 6
ff_embd = 1536
ff_layers = 1

# ---------- Training hyperparams :

lr_max = 3e-4
lr_min = 1e-5
warmup_steps = 375
max_steps = 7500
checkpoint=4000

# cosine lr with warmup
def get_lr(step): # we need to implement the cosine learning rate based on the step, the formula is the following :
    # lr(t) = lr_min +1/2(lr_max-lr_min)(1+cos(pi*t/T)) + warmup branch
    if step <= warmup_steps:
        return ((lr_max-lr_min)/(warmup_steps))*step + lr_min
    else:
        return lr_min + (1/2)*(lr_max - lr_min)*(1+math.cos((math.pi * (step-warmup_steps))/(max_steps-warmup_steps)))

# ------- Importing our tiny ass dataset






ds = pass 
loader = DataLoader(ds, batch_size=B, shuffle=True, drop_last=True) #shuffles indeces (SGD)
dev_loader  = DataLoader(ds_dev,  batch_size=B, shuffle=False, drop_last=True)
test_loader = DataLoader(ds_test, batch_size=B, shuffle=False, drop_last=True)

# -------- model and optimizer



model = ViT_model(B=B, H=H, W=W, n_head=n_head, ff_layers=ff_layers, ff_embd=ff_embd, n_blocks=n_blocks, div_H=div_H, div_W=div_W, n_embd=n_embd)
model.to(device)
# passing in all of our model hyperparameters

bb   = [p for n, p in model.named_parameters() if n.startswith("conv_layer.backbone")]
rest = [p for n, p in model.named_parameters() if not n.startswith("conv_layer.backbone")]
groups = [
    {"params": [p for p in bb   if p.dim() >= 2], "weight_decay": 0.05, "scale": 0.1},
    {"params": [p for p in bb   if p.dim() <  2], "weight_decay": 0.0,  "scale": 0.1},
    {"params": [p for p in rest if p.dim() >= 2], "weight_decay": 0.05, "scale": 1.0},
    {"params": [p for p in rest if p.dim() <  2], "weight_decay": 0.0,  "scale": 1.0}, # different groups
]

opt = torch.optim.AdamW( # classic AdamW ossiclation fighting + weight decay on non-layernorn/bias
    groups,
    lr=3e-4, betas=(0.9, 0.999))

# btw, here since faced with some continuous and discrete data, we shall use a mix of loss functions to account for that on each feature


# ----- Loss function hyperparameters




def mixed_loss_function(x, y):
    # we should account for binary, dicrete and continuous values of our tensors
    pass

# helper we'll use to to evaluate our dev and test losses
@torch.no_grad()
def evaluate(loader):
    model.eval()
    total, n = 0.0, 0
    for frame, y in loader:
        frame, y = frame.to(device), y.to(device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            total += mixed_loss_function(model(frame), y).item()
        n += 1
    model.train()
    return total / n



model.compile() #compiling for faster exectution


it = iter(loader) # iterator over our ds

# now the training loop

for step in range(max_steps):
    try:
        frame, y = next(it)
    except StopIteration:
        it = iter(loader)
        frame, y = next(it)
    frame, y = frame.to(device), y.to(device) #making sure no bs happens with the device type
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        pred = model(frame) 
        loss = mixed_loss_function(pred, y) #y^, y
    loss.backward() #+='ing on the gradients
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # gradient clipping to fight overfitting
    lr = get_lr(step) #scheduled lr
    for g in opt.param_groups:
        g["lr"] = lr * g.get("scale", 1.0) # using our actual lr into the AdamW's EMA formula
    opt.step()
    opt.zero_grad(set_to_none=True)
    #print/logging train loss at every step
    print(f"Step : {step}, Training Loss : {loss.item():.4f}")
    log.write(f"train,{step},{lr:.2e},{loss.item():.4f}\n"); log.flush()

    # every 250 step, evaluate dev, loss
    if step % 250 == 0 or step == max_steps - 1:
        dev_loss = evaluate(dev_loader)
        print(f"Dev Loss : {dev_loss:.4f}")
        log.write(f"dev,{step},{lr:.2e},{dev_loss:.4f}\n"); log.flush()

    # every 500 step, sample output from the model
    if step % 500 == 0 or step == max_steps - 1:
        torch.save({"pred": pred.detach().float().cpu(), "target": y.cpu()},
           f"{run}/samples/step_{step:06d}.pt")

    # at checkpoints and final iter, save the weights
    if step % checkpoint == 0 or step == max_steps - 1:
        torch.save({"step": step, "model": model.state_dict(), "optimizer": opt.state_dict()},
                f"{run}/checkpoints/step_{step:07d}.pt")

test_loss = evaluate(test_loader)
print(f"Test Loss : {test_loss:.4f}")
log.write(f"test,{step},{lr:.2e},{test_loss:.4f}\n"); log.flush()
# finally, evaluate our test loss