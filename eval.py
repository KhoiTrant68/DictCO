import argparse
import math
import sys
import time
import warnings
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
from pytorch_msssim import ms_ssim

# Assuming DCAE is in a local file 'models.py'
from models.dcae import DCAE

warnings.filterwarnings("ignore")

# --- Helper Functions ---

def compute_psnr(a, b):
    mse = torch.mean((a - b)**2).item()
    if mse == 0:
        return 100
    return -10 * math.log10(mse)

def compute_msssim(a, b):
    # Ensure data_range is 1.0 (since images are normalized 0-1)
    return -10 * math.log10(1 - ms_ssim(a, b, data_range=1.0).item())

def compute_bpp_estimated(out_net):
    size = out_net['x_hat'].size()
    num_pixels = size[0] * size[2] * size[3]
    bpp = 0
    # Sum likelihoods from all entropy bottlenecks
    for likelihoods in out_net['likelihoods'].values():
        bpp += torch.log(likelihoods).sum() / (-math.log(2) * num_pixels)
    return bpp.item()

def pad(x, p=128):
    h, w = x.size(2), x.size(3)
    new_h = (h + p - 1) // p * p
    new_w = (w + p - 1) // p * p
    
    padding_left = (new_w - w) // 2
    padding_right = new_w - w - padding_left
    padding_top = (new_h - h) // 2
    padding_bottom = new_h - h - padding_top
    
    padding_dims = (padding_left, padding_right, padding_top, padding_bottom)
    
    x_padded = F.pad(
        x,
        padding_dims,
        mode="constant",
        value=0,
    )
    return x_padded, padding_dims

def crop(x, padding):
    return F.pad(
        x,
        (-padding[0], -padding[1], -padding[2], -padding[3]),
    )

def parse_args(argv):
    parser = argparse.ArgumentParser(description="DCAE Evaluation Script")
    parser.add_argument("--cuda", action="store_true", help="Use CUDA")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--data", type=str, required=True, help="Path to dataset folder")
    parser.add_argument("--save_path", type=str, default=None, help="Path to save outputs")
    parser.add_argument("--real", action="store_true", help="Enable actual bit-stream compression")
    return parser.parse_args(argv)

# --- Main Logic ---

def main(argv):
    args = parse_args(argv)
    
    # Optimization: Enable cuDNN benchmark for faster convolutions
    if args.cuda and torch.cuda.is_available():
        device = 'cuda:0'
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
    else:
        device = 'cpu'

    # Setup Paths
    data_path = Path(args.data)
    save_path = Path(args.save_path) if args.save_path else None
    
    if save_path:
        save_path.mkdir(parents=True, exist_ok=True)

    # Filter Images
    extensions = {'.jpg', '.png', '.jpeg'}
    img_list = [f for f in data_path.iterdir() if f.suffix.lower() in extensions]
    
    if not img_list:
        print(f"No images found in {data_path}")
        return

    # Load Model
    print(f"Loading model from {args.checkpoint}...")
    net = DCAE().to(device)
    
    # Load Checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    state_dict = {k.replace("module.", ""): v for k, v in checkpoint["state_dict"].items()}
    net.load_state_dict(state_dict)
    
    net.eval()

    # If doing real compression, update entropy bottlenecks
    if args.real:
        print("Updating entropy bottlenecks...")
        net.update()

    # Metrics Accumulators
    metrics = {
        "psnr": 0.0,
        "ms_ssim": 0.0,
        "bpp": 0.0,
        "time_total": 0.0,
        "time_enc": 0.0,
        "time_dec": 0.0
    }
    
    count = 0
    to_tensor = transforms.ToTensor()

    print(f"Starting inference on {len(img_list)} images...")
    
    # [Image of deep learning image compression architecture]
    # The loop below handles the flow: Image -> Pad -> Encoder -> Quantizer -> Decoder -> Crop

    for img_file in img_list:
        # Load Image
        img = Image.open(img_file).convert('RGB')
        x = to_tensor(img).unsqueeze(0).to(device)
        
        # Pad
        x_padded, padding = pad(x, p=128)
        x_padded = x_padded.to(device) # Ensure padded tensor is on device

        num_pixels = x.size(0) * x.size(2) * x.size(3)
        
        # Inference
        with torch.no_grad():
            if args.real:
                # --- Real Compression (Encode -> String -> Decode) ---
                
                # Encode
                if args.cuda: torch.cuda.synchronize()
                t_start = time.time()
                out_enc = net.compress(x_padded)
                if args.cuda: torch.cuda.synchronize()
                t_enc = time.time() - t_start
                
                # Decode
                if args.cuda: torch.cuda.synchronize()
                t_start = time.time()
                out_dec = net.decompress(out_enc["strings"], out_enc["shape"])
                if args.cuda: torch.cuda.synchronize()
                t_dec = time.time() - t_start
                
                # Processing
                x_hat = crop(out_dec["x_hat"], padding)
                x_hat.clamp_(0, 1)
                
                # Metrics
                total_bits = sum(len(s[0]) for s in out_enc["strings"]) * 8.0
                current_bpp = total_bits / num_pixels
                
                metrics["time_enc"] += t_enc
                metrics["time_dec"] += t_dec
                metrics["time_total"] += (t_enc + t_dec)
                
            else:
                # --- Estimation (Forward Pass) ---
                
                if args.cuda: torch.cuda.synchronize()
                t_start = time.time()
                out_net = net(x_padded)
                if args.cuda: torch.cuda.synchronize()
                t_total = time.time() - t_start
                
                # Processing
                x_hat = crop(out_net["x_hat"], padding)
                x_hat.clamp_(0, 1)
                
                # Metrics
                current_bpp = compute_bpp_estimated(out_net)
                metrics["time_total"] += t_total

        # Compute Image Quality Metrics (Once per image)
        current_psnr = compute_psnr(x, x_hat)
        current_msssim = compute_msssim(x, x_hat)
        
        # Update Accumulators
        count += 1
        metrics["psnr"] += current_psnr
        metrics["ms_ssim"] += current_msssim
        metrics["bpp"] += current_bpp
        
        # Print Status
        print(f"[{count}/{len(img_list)}] {img_file.name} | "
              f"Bpp: {current_bpp:.3f} | PSNR: {current_psnr:.2f} | MS-SSIM: {current_msssim:.2f}")

        # Save Artifacts (if requested)
        if save_path:
            # Save Image
            save_image(x_hat, save_path / f"recon_{img_file.name}")
            
            # Save Metrics Text
            with open(save_path / f"metrics_{img_file.stem}.txt", 'w') as f:
                f.write(f'PSNR: {current_psnr:.2f}dB\n')
                f.write(f'Bitrate: {current_bpp:.3f}bpp\n')
                f.write(f'MS-SSIM: {current_msssim:.4f}\n')

    # --- Final Averages ---
    print("-" * 40)
    print(f"Results ({count} images):")
    print(f"Avg PSNR:      {metrics['psnr'] / count:.2f} dB")
    print(f"Avg MS-SSIM:   {metrics['ms_ssim'] / count:.4f}")
    print(f"Avg Bitrate:   {metrics['bpp'] / count:.3f} bpp")
    
    if args.real:
        print(f"Avg Encode Time: {metrics['time_enc'] / count * 1000:.2f} ms")
        print(f"Avg Decode Time: {metrics['time_dec'] / count * 1000:.2f} ms")
    
    print(f"Avg Total Time:  {metrics['time_total'] / count * 1000:.2f} ms")

if __name__ == "__main__":
    main(sys.argv[1:])