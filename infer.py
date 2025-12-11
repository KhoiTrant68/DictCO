import argparse
import math
import struct
import sys
import time
import warnings
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image

# Import DCAE model from local module
from models.dcae import DCAE

# Suppress specific warnings if necessary, but keep critical ones
warnings.filterwarnings("ignore")


def pad(x, p):
    """Pads the input tensor to be divisible by p."""
    h, w = x.size(2), x.size(3)
    new_h = (h + p - 1) // p * p
    new_w = (w + p - 1) // p * p

    padding_left = (new_w - w) // 2
    padding_right = new_w - w - padding_left
    padding_top = (new_h - h) // 2
    padding_bottom = new_h - h - padding_top

    x_padded = F.pad(
        x,
        (padding_left, padding_right, padding_top, padding_bottom),
        mode="constant",
        value=0,
    )
    return x_padded, (padding_left, padding_right, padding_top, padding_bottom)


def crop(x, padding):
    """Crops the tensor back to original size."""
    return F.pad(
        x,
        (-padding[0], -padding[1], -padding[2], -padding[3]),
    )


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Image Compression Script")
    parser.add_argument("--cuda", action="store_true", help="Use cuda")
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to a checkpoint"
    )
    parser.add_argument(
        "--save_path", type=str, required=True, help="Path to save output"
    )
    parser.add_argument("--data", type=str, required=True, help="Path to dataset")
    parser.add_argument(
        "--mode", type=str, choices=["compress", "decompress"], required=True
    )
    parser.add_argument(
        "--patch_size", type=int, default=128, help="Padding size divisor"
    )
    return parser.parse_args(argv)


def save_bin(strings, size, img_name, save_dir):
    """Saves compressed strings and metadata to a binary file."""
    save_dir = Path(save_dir) / "bin"
    save_dir.mkdir(parents=True, exist_ok=True)

    bin_path = save_dir / (Path(img_name).stem + ".bin")

    # strings is usually a list of lists -> [[y_string], [z_string]]
    s_y = strings[0][0]
    s_z = strings[1][0]

    with open(bin_path, "wb") as f:
        # Save Original Height/Width (Unsigned Short)
        f.write(struct.pack(">H", size[0]))
        f.write(struct.pack(">H", size[1]))

        # Save Length of Y string (Unsigned Int) and data
        f.write(struct.pack(">I", len(s_y)))
        f.write(s_y)

        # Save Length of Z string (Unsigned Int) and data
        f.write(struct.pack(">I", len(s_z)))
        f.write(s_z)


def read_bin(bin_path):
    """Reads compressed data from binary file."""
    with open(bin_path, "rb") as f:
        h = struct.unpack(">H", f.read(2))[0]
        w = struct.unpack(">H", f.read(2))[0]

        len_y = struct.unpack(">I", f.read(4))[0]
        str_y = f.read(len_y)

        len_z = struct.unpack(">I", f.read(4))[0]
        str_z = f.read(len_z)

    # Re-calculate padding based on stored H/W
    # Note: Using default 128 here, ensure this matches the 'compress' stage
    padding_size, padding = calculate_padding(h, w)

    # Calculate latent shape (usually stride 64 for many compression models)
    z_shape = [padding_size[0] // 64, padding_size[1] // 64]

    string = [[str_y], [str_z]]
    return string, z_shape, padding


def calculate_padding(h, w, p=128):
    new_h = (h + p - 1) // p * p
    new_w = (w + p - 1) // p * p

    padding_left = (new_w - w) // 2
    padding_right = new_w - w - padding_left
    padding_top = (new_h - h) // 2
    padding_bottom = new_h - h - padding_top

    return (new_h, new_w), (padding_left, padding_right, padding_top, padding_bottom)


def main(argv):
    args = parse_args(argv)

    # --- Optimization 1: Enable cuDNN for speed ---
    if args.cuda and torch.cuda.is_available():
        device = "cuda:0"
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
    else:
        device = "cpu"

    # Setup paths
    data_path = Path(args.data)
    save_path = Path(args.save_path)

    # Load Model
    print(f"Loading model from {args.checkpoint}...")
    net = DCAE().to(device)

    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location=device)
        # Handle 'module.' prefix if model was trained with DataParallel
        state_dict = {
            k.replace("module.", ""): v for k, v in checkpoint["state_dict"].items()
        }
        net.load_state_dict(state_dict)

    net.eval()
    net.update()  # Update entropy bottleneck (crucial for compression)

    print(f"Starting {args.mode} mode on {device}...")

    # Define transforms once
    to_tensor = transforms.ToTensor()

    # Filter files
    if args.mode == "compress":
        extensions = {".jpg", ".png", ".jpeg"}
        files = [f for f in data_path.iterdir() if f.suffix.lower() in extensions]
    else:
        files = [f for f in data_path.iterdir() if f.suffix.lower() == ".bin"]

    # Main Loop
    with torch.no_grad():
        for file_path in files:
            img_name = file_path.name

            if args.mode == "compress":
                # Load image
                img = Image.open(file_path).convert("RGB")
                x = to_tensor(img).unsqueeze(0).to(device)

                x_size = x.size()[-2:]  # H, W
                x_padded, padding = pad(x, args.patch_size)

                # --- Bug Fix: Assign the result of .to(device) ---
                x_padded = x_padded.to(device)

                # Inference
                out_enc = net.compress(x_padded)

                # Save
                save_bin(out_enc["strings"], x_size, img_name, save_path)

            elif args.mode == "decompress":
                # Read Bin
                string, shape, padding = read_bin(file_path)

                # Inference
                out_dec = net.decompress(string, shape)

                # Post-process
                x_hat = crop(out_dec["x_hat"], padding)
                x_hat.clamp_(0, 1)  # In-place clamp for slight memory speedup

                # --- Optimization 2: Faster Saving ---
                output_file = save_path / (file_path.stem + ".png")
                output_file.parent.mkdir(parents=True, exist_ok=True)
                save_image(x_hat, output_file)
                print(f"Saved: {output_file}")


if __name__ == "__main__":
    main(sys.argv[1:])
