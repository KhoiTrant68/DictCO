## Project Structure

```
DictCO/
├── train.py          # Model training pipeline
├── eval.py           # Evaluation and metrics computation
├── infer.py          # Inference (compression/decompression)
├── test.py           # Unit tests
├── models/           # Model definitions
├── modules/          # Module implementations
├── docs/             # Documentation (empty)
├── LICENSE           # Apache 2.0 License
└── README.md         # Project description (minimal)
```

---

## Detailed Code Analysis

### 1. **train.py** (13,637 bytes)
**Purpose**: Training pipeline for the DCAE model with distributed training support.

**Key Components:**

- **AverageMeter Class**: Tracks running averages of metrics during training
  - Methods: `reset()`, `update()`, `avg`, `val`, `sum`

- **RateDistortionLoss Class**: Custom loss function for image compression
  ```
  Loss = λ × distortion_loss + bpp_loss
  ```
  - Supports two distortion metrics: MSE (Mean Squared Error) or MS-SSIM (Multi-Scale Structural Similarity)
  - Balances image quality (distortion) with compression efficiency (bits-per-pixel)

- **configure_optimizers()**: Separates parameters into main and auxiliary optimizers
  - Main optimizer: Standard model parameters
  - Auxiliary optimizer: Quantile parameters (for entropy modeling)

- **train_one_epoch()**: Single training epoch handler
  - Implements Automatic Mixed Precision (AMP) using `GradScaler`
  - Features gradient clipping and non-blocking GPU transfers
  - Logs metrics every 100 iterations

- **test_epoch()**: Validation loop
  - Computes loss, BPP loss, and distortion metrics
  - Returns average validation loss for checkpointing

- **Distributed Training Features**:
  - Supports multi-GPU training via `DistributedDataParallel`
  - Uses `DistributedSampler` for proper data distribution
  - Rank-based checkpoint saving (only rank 0 process saves)
  - TensorBoard logging integration

**Optimization Techniques:**
- cuDNN benchmark enabled for fixed input sizes
- Non-blocking GPU transfers
- `set_to_none=True` in `zero_grad()` for faster memory clearing
- Learning rate scheduling with MultiStepLR

---

### 2. **eval. py** (7,754 bytes)
**Purpose**: Evaluation script that computes image quality and compression metrics.

**Key Functions:**

- **compute_psnr(a, b)**: Peak Signal-to-Noise Ratio
  - Formula: PSNR = -10 × log₁₀(MSE)
  - Returns 100 for perfect reconstructions (MSE=0)

- **compute_msssim(a, b)**: MS-SSIM metric (logarithmic scale)
  - Multi-scale structural similarity assessment
  - Better perceptual quality metric than PSNR

- **compute_bpp_estimated()**: Bits-Per-Pixel calculation
  - Sums negative log-likelihoods from entropy bottlenecks
  - Estimates compression ratio without actual encoding

- **pad() / crop()**: Image padding utilities
  - Pads to multiples of 128 pixels (required for some architectures)
  - Centers the original image in padded region
  - Crops back after inference

**Two Evaluation Modes:**

1. **Estimation Mode** (default): Fast forward pass
   - Calculates BPP from likelihood estimates
   - No actual bit-stream encoding

2. **Real Compression** (--real flag): Actual encoding/decoding
   - Calls `net.compress()` for entropy-coded strings
   - Calls `net.decompress()` to reconstruct
   - Measures actual encode/decode times

**Output Metrics:**
- PSNR (dB)
- MS-SSIM (quality)
- BPP (bits per pixel)
- Encode/Decode times (ms)

---

### 3. **infer.py** (6,646 bytes)
**Purpose**: Compression/decompression inference script with binary file I/O.

**Key Features:**

- **Two Operation Modes**:
  1.  **Compress Mode**: Image → Padded → Encoded → Binary file
  2. **Decompress Mode**: Binary file → Decoded → Padded image → Output

- **Binary File Format** (custom):
  ```
  [2 bytes: height] [2 bytes: width]
  [4 bytes: len_y] [variable: y_string]
  [4 bytes: len_z] [variable: z_string]
  ```
  - Stores original dimensions for proper reconstruction
  - Separates y and z entropy strings

- **Functions**:
  - `save_bin()`: Serializes compressed data with metadata
  - `read_bin()`: Deserializes binary files
  - `calculate_padding()`: Determines padding amounts for given dimensions
  - `pad() / crop()`: Image transformation utilities

**Optimization Details:**
- cuDNN benchmarking enabled
- Entropy bottleneck updating via `net.update()`
- In-place operations (`clamp_()`) for memory efficiency
- Batch processing of multiple files

---


## Architecture & Workflow

### Image Compression Pipeline:

```
Input Image (H×W)
    ↓
Padding (to 128 multiple) → (H'×W')
    ↓
Encoder (DCAE) → Quantizer
    ↓
Entropy Model (y, z bottlenecks)
    ↓
Arithmetic Coding → Bit-stream
    ↓
Stored as Binary File
```

### Decompression Pipeline:

```
Binary File (Bit-stream)
    ↓
Entropy Decoding → Quantized features
    ↓
Decoder (DCAE)
    ↓
Post-processing (clamp, crop)
    ↓
Reconstructed Image
```

---

## Technical Highlights

### 1. **Rate-Distortion Trade-off**
- Lagrangian parameter (λ) controls compression vs. quality
- Two loss variants: MSE-based or MS-SSIM-based

### 2. **Distributed Training**
- Multi-GPU support with proper synchronization
- Rank-based logging (prevents duplicate outputs)
- Checkpointing strategy for resumable training

### 3. **Entropy Modeling**
- Separate y and z entropy bottlenecks
- Auxiliary loss for training quantile parameters
- Real bit-stream compression capability

### 4. **Performance Optimizations**
- Automatic Mixed Precision (AMP) with GradScaler
- Non-blocking GPU memory transfers
- cuDNN benchmarking for fixed-size inputs
- In-place tensor operations where possible


## 📁 Directory Structure

```
DictCO/
├── models/
│   └── dcae.py           (17,176 bytes) - Main compression model
├── modules/
│   ├── resnet_module.py  (2,602 bytes)  - ResNet building blocks
│   └── swin_module.py    (19,803 bytes) - Swin Transformer components
```

---

# MODULES FOLDER ANALYSIS

## 1. **modules/resnet_module.py** (2,602 bytes)

### Purpose
Provides optimized residual bottleneck blocks for downsampling/upsampling in the encoder/decoder pathways. 

### Classes

#### **ResidualBottleneckBlock**
```
Architecture: Conv1x1 → BN → ReLU → Conv3x3 → BN → ReLU → Conv1x1 → BN
             (with residual connection and optional downsampling)
```

**Key Features:**
- **Expansion Factor**: 4× (compresses channel dimension in middle layer)
- **Bottleneck Design**: `in_ch → out_ch/4 → out_ch/4 → out_ch`
- **Residual Connection**: Identity shortcut with optional `1×1 Conv + BN` for channel/spatial mismatch
- **Batch Normalization**: Standard ResNet-style BN placement
- **ReLU Activation**: Inplace operations for memory efficiency

**Example Flow** (e.g., in_ch=256, out_ch=256):
```
Input [B, 256, H, W]
  ↓
Conv1x1 (256→64) + BN + ReLU
  ↓
Conv3x3 (64→64) + BN + ReLU
  ↓
Conv1x1 (64→256) + BN
  ↓
Add Identity + ReLU
  ↓
Output [B, 256, H, W]
```

---

#### **ResidualBottleneckBlockWithStride**
Downsampling block for analysis (encoder) pathway. 

**Architecture:**
```
Input [B, in_ch, H, W]
  ↓
Conv2d (kernel=5, stride=2, padding=2) → Spatial: H→H/2, W→W/2
  ↓
3× ResidualBottleneckBlock (maintains channel count)
  ↓
Output [B, out_ch, H/2, W/2]
```

**Usage in DCAE:**
```python
ResidualBottleneckBlockWithStride(96, 144)    # 96ch → 144ch, H/2
ResidualBottleneckBlockWithStride(144, 256)   # 144ch → 256ch, H/2
```

---

#### **ResidualBottleneckBlockWithUpsample**
Upsampling block for synthesis (decoder) pathway.

**Architecture:**
```
Input [B, in_ch, H, W]
  ↓
3× ResidualBottleneckBlock (maintains in_ch)
  ↓
ConvTranspose2d (kernel=5, stride=2, padding=2, output_padding=1)
  → Spatial: H→2H, W→2W
  ↓
Output [B, out_ch, 2H, 2W]
```

**Usage in DCAE:**
```python
ResidualBottleneckBlockWithUpsample(256, 144)  # 256ch → 144ch, H×2
ResidualBottleneckBlockWithUpsample(144, 96)   # 144ch → 96ch, H×2
```

---

## 2. **modules/swin_module.py** (19,803 bytes)

### Purpose
Implements Swin Transformer blocks and novel cross-attention mechanisms for feature processing.

### Core Components

#### **WMSA (Window Multi-head Self-Attention)**
Optimized window-based multi-head self-attention with relative position bias. 

**Key Innovations:**
1. **Pre-computed Relative Position Indices**: Stored as buffer (no CPU-GPU sync in forward)
2. **Window Partitioning**: Limits attention to local windows (e.g., 8×8)
3. **Cyclic Shift**: For SW (Shifted Window) mode to capture cross-window context

**Forward Flow:**
```
Input [B, H, W, C]
  ↓
[Optional] Cyclic shift (for SW mode)
  ↓
Partition into windows: [B, num_windows, window_size², C]
  ↓
Linear projection to Q, K, V (3×C)
  ↓
Multi-head split: [H, B, num_windows, window_size², C/H]
  ↓
Attention with relative position bias
  ↓
[Optional] Cyclic shift back (for SW mode)
  ↓
Output [B, H, W, C]
```

**Attention Equation:**
```
Attention(Q, K, V) = softmax((Q·K^T / √d) + RelativePosBias) · V
```

---

#### **ResScaleConvGateBlock**
Combines window attention with learnable residual scaling.

**Architecture:**
```
Input [B, H, W, C]
  ↓
Residual Branch 1: LayerNorm → WMSA → DropPath → Scale
  ↓ (add)
Residual Branch 2: LayerNorm → ConvGELU MLP → DropPath → Scale
  ↓ (add)
Output [B, H, W, C]
```

**Key Idea**: Learnable scale parameters (initialized to 1.0) allow dynamic importance weighting of residual connections.

---

#### **SwinBlockWithConvMulti**
Full Swin block with multiple transformer layers and final convolution.

**Processing:**
```
Input [B, C, H, W]
  ↓
Permute to [B, H, W, C]
  ↓
Pad to window multiple (if needed)
  ↓
Alternating W and SW blocks:
  - Block 0: Window attention (W type)
  - Block 1: Shifted window attention (SW type)
  - Block 2: Window attention
  - ...  (configurable)
  ↓
Permute back to [B, C, H, W]
  ↓
Conv2d (3×3, stride=1) for feature transformation
  ↓
Remove padding
  ↓
Residual: Output + Input
  ↓
Output [B, C, H, W]
```

---

#### **DWConv (Depth-wise Convolution)**
Efficient convolution with group convolution (groups=channels).

```
Input [B, C, H, W]
  ↓
Conv2d(kernel=3, stride=1, padding=1, groups=C)
  ↓
Output [B, C, H, W]
```

**Efficiency**: O(C·k²) instead of O(C²·k²) for regular convolution.

---

#### **ConvGELU**
MLP block using depthwise convolution with GELU activation.

**Architecture:**
```
Input [B, *, C]
  ↓
Linear: C → 2×hidden
  ↓
Split into 2 branches: [B, *, hidden], [B, *, hidden]
  ↓
Branch 1 → DWConv → GELU (gating)
Branch 2 → (value)
  ↓
Element-wise multiply: gated × value
  ↓
Linear: hidden → out_features
  ↓
Output [B, *, out_features]
```

---

#### **MultiScaleAggregation (MSFA)**
Extracts local texture context via dense connections and spatial attention.

**Architecture:**
```
Input [B, H, W, C]
  ↓
Permute to [B, C, H, W]
  ↓
Conv1×1: C → C
  ↓
DenseBlock (3 layers of ConvWithDW + concatenation)
  ↓
SpatialAttentionModule:
  - Mean pooling over channels → [B, 1, H, W]
  - Max pooling over channels → [B, 1, H, W]
  - Concatenate → [B, 2, H, W]
  - Conv + Sigmoid → [B, 1, H, W]
  ↓
Scale: feature_map × spatial_attention
  ↓
Permute back to [B, H, W, C]
  ↓
Output: Content-weighted features
```

---

#### **MutiScaleDictionaryCrossAttentionGELU** (Baseline)
Cross-attention using a single global dictionary for entropy modeling.

**Flow:**
```
Query from x:          x → Linear → [B, H, W, dict_dim]
                          ↓
                       MSFA aggregation
                          ↓
                       Query projection → Q [B, Heads, H×W, head_dim]

Dictionary (dt):      dt → LayerNorm → Linear → K [B, Heads, dict_entries, head_dim]
                                            ↓
                                            V [B, Heads, dict_entries, head_dim]

Cross Attention:      Attention(Q, K, V) = softmax(Q·K^T / √d) · V
                      Output: [B, H, W, dict_dim]
                      ↓
                      Linear projection → [B, C, H, W]
```

---

#### **MoEDictionaryCrossAttention** ⭐ (Novel Contribution)
**Mixture-of-Experts** variant with K specialized dictionaries and spatial routing.

**Key Innovation**: Instead of one global dictionary, uses K expert dictionaries with a learned router.

**Architecture:**

```
1. Expert Bank:
   self. experts: [K, N, dict_dim]
   K = num_experts (e.g., 4)
   N = expert_entries (e.g., 64)
   dict_dim = 640 (32 * 20 heads)

2. Spatial Router:
   Takes local features x: [B, H, W, dict_dim]
   Outputs routing logits: [B, H, W, K]
   Applies softmax → routing_weights: [B, H, W, K]
   Each spatial location gets K expert weights

3. Expert Attention Loop (for each of K experts):
   - Expert K dictionary: [N, dict_dim]
   - Query from x: [B, Heads, H×W, head_dim]
   - Compute attention: Q · K^T → [B, Heads, H×W, N]
   - Aggregate values: probs · V → [B, H, W, dict_dim]
   - Weight by routing gate: expert_out × gate_k
   
4. Final Output:
   Σ(all_k: expert_out_k × gate_k)
   + MLP + Residuals
   → [B, C, H, W]
```

**Pseudo-code:**
```python
routing_logits = self.router(x)  # [B, H, W, K]
routing_weights = softmax(routing_logits)  # [B, H, W, K]

final_context = 0
for k in range(num_experts):
    expert_dict = self.experts[k]  # [N, dict_dim]
    
    # Attention
    sim = Q @ K(expert_dict) / √d  # [B, H×W, N]
    probs = softmax(sim)
    expert_out = probs @ V(expert_dict)  # [B, H, W, dict_dim]
    
    # Gating
    gate = routing_weights[.. ., k]  # [B, H, W]
    final_context += expert_out * gate. unsqueeze(-1)
```

**Advantages over baseline:**
- ✅ **Adaptive Expert Selection**: Different experts specialize in different image regions
- ✅ **Increased Capacity**: K× more dictionary entries without K× computation (sparse gating)
- ✅ **Better Context Modeling**: Spatial router respects local image statistics
- ✅ **Improved Compression**: Better entropy coding through diverse dictionary sources

---

# MODELS FOLDER ANALYSIS

## **models/dcae.py** (17,176 bytes)

### DCAE: Deep Compression AutoEncoder

**Overall Architecture:**

```
┌─────────────────────────────────────────────────────────────────┐
│                        FORWARD FLOW                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Image Input [B, 3, H, W]                                       │
│       ↓                                                          │
│  ╔═══════════════════════════════════════════════════════════╗ │
│  ║ ANALYSIS TRANSFORM (g_a) - Encoder                      ║ │
│  ║                                                          ║ │
│  ║ ResBottleneck(3→96, stride=2)        [B, 96, H/2, W/2] ║ │
│  ║ SwinBlock + ResBottleneck(96→144)    [B, 144, H/4, W/4]║ │
│  ║ SwinBlock + ResBottleneck(144→256)   [B, 256, H/8, W/8]║ │
│  ║ SwinBlock + Conv(256→320, stride=2)  [B, 320, H/16,W/16║ │
│  ╚═══════════════════════════════════════════════════════════╝ │
│       ↓                                                          │
│  Latent y: [B, 320, H/16, W/16]                                │
│       ↓                                                          │
│  ╔═══════════════════════════════════════════════════════════╗ │
│  ║ HYPER-ANALYSIS (h_a)                                     ║ │
│  ║                                                          ║ │
│  ║ ResBottleneck(320→192, stride=2)     [B, 192, H/32, W/32║ │
│  ║ SwinBlock(192→192)                   [B, 192, H/32, W/32║ │
│  ║ Conv(192→192, stride=2)              [B, 192, H/64, W/64║ │
│  ╚═══════════════════════════════════════════════════════════╝ │
│       ↓                                                          │
│  Hyper-latent z: [B, 192, H/64, W/64]                          │
│       ↓                                                          │
│  ╔═══════════════════════════════════════════════════════════╗ │
│  ║ ENTROPY BOTTLENECK                                       ║ │
│  ║ (Quantization with likelihood estimation)               ║ │
│  ╚═══════════════════════════════════════════════════════════╝ │
│       ↓                                                          │
│  z_hat: [B, 192, H/64, W/64] (quantized)                       │
│       ↓                                                          │
│  ╔═══════════════════════════════════════════════════════════╗ │
│  ║ HYPER-SYNTHESIS (h_z_s1, h_z_s2)                        ║ │
│  ║                                                          ║ │
│  ║ Path 1: Upsample z_hat → latent_scales                 ║ │
│  ║ Path 2: Upsample z_hat → latent_means                  ║ │
│  ║ Both: [B, 320, H/16, W/16]                             ║ │
│  ╚═══════════════════════════════════════════════════════════╝ │
│       ↓                                                          │
│  ╔═══════════════════════════════════════════════════════════╗ │
│  ║ SLICE-WISE ENTROPY MODELING                             ║ │
│  ║ (Iterative refinement of y latents)                     ║ │
│  ║                                                          ║ │
│  ║ Split y into 5 slices (along channel dim)              ║ │
│  ║                                                          ║ │
│  ║ For each slice i:                                       ║ │
│  ║   1. MoE Cross-Attention (dt_cross_attention[i])        ║ │
│  ║      - Query: [latent_scales, latent_means] + prev_i-1 ║ │
│  ║      - Output: dict_info                                ║ │
│  ║                                                          ║ │
│  ║   2. Mean/Scale Prediction (cc_mean/scale_transforms)  ║ │
│  ║      - Input: [query, dict_info]                        ║ │
│  ║      - Outputs: mu_i, scale_i                           ║ │
│  ║                                                          ║ │
│  ║   3. Gaussian Conditional Encoding                      ║ │
│  ║      - Model: y_slice ~ N(mu_i, scale_i)               ║ │
│  ║      - Quantize: y_hat_slice = STE_round(y_slice - mu) ║ │
│  ║                                                          ║ │
│  ║   4. Laplace Residual Prediction (lrp_transforms)      ║ │
│  ║      - Refine: y_hat_slice += 0.5 * tanh(lrp)          ║ │
│  ║      - Improves reconstruction quality                  ║ │
│  ╚═══════════════════════════════════════════════════════════╝ │
│       ↓                                                          │
│  Refined y_hat: [B, 320, H/16, W/16]                           │
│       ↓                                                          │
│  ╔═══════════════════════════════════════════════════════════╗ │
│  ║ SYNTHESIS TRANSFORM (g_s) - Decoder                    ║ │
│  ║                                                          ║ │
│  ║ ConvTranspose(320→256, stride=2)     [B, 256, H/8, W/8]║ │
│  ║ SwinBlock + ResUpsample(256→144)     [B, 144, H/4, W/4]║ │
│  ║ SwinBlock + ResUpsample(144→96)      [B, 96, H/2, W/2] ║ │
│  ║ SwinBlock + ResUpsample(96→3)        [B, 3, H, W]      ║ │
│  ╚═══════════════════════════════════════════════════════════╝ │
│       ↓                                                          │
│  Reconstructed Image x_hat: [B, 3, H, W] (clipped to [0, 1])   │
│       ↓                                                          │
│  Return: {x_hat, likelihoods (y, z), para (means, scales)}     │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Initialization Details

```python
DCAE(
    head_dim=[8, 16, 32, 32, 16, 8],     # Heads per Swin block
    N=192,                                # Hyper-latent channels
    M=320,                                # Main latent channels
    num_slices=5,                         # Entropy slices
    max_support_slices=5,                 # Previous slices available
)
```

**Feature Dimensions:**
```python
feature_dim = [96, 144, 256]
↓
Encoder levels: 96 → 144 → 256 → 320 (M)
Decoder levels: 320 → 256 → 144 → 96 → 3 (RGB)
```

---

### Key Methods

#### **1. forward(x) - Inference Mode**

Returns likelihood-based metrics suitable for training. 

```python
return {
    "x_hat": reconstructed_image,
    "likelihoods": {
        "y": y_likelihoods,    # For rate loss
        "z": z_likelihoods     # For rate loss
    },
    "para": {
        "means": means,        # Entropy model parameters
        "scales": scales,      # Entropy model parameters
        "y": y                 # Original latents
    }
}
```

**Loss Computation in training. py:**
```python
loss = λ × distortion(x_hat, x) + bits_per_pixel(likelihoods)
```

---

#### **2. compress(x) - Real Bit-stream Encoding**

Produces actual entropy-coded binary strings.

**Process:**
1. Encode y and z latents through entropy bottlenecks
2. Use **Arithmetic Coding** (RANS - Asymmetric Numeral Systems)
3. Returns:
   ```python
   {
       "strings": [[y_string], z_strings],  # Binary bitstreams
       "shape": z. shape[-2:]                 # Latent shape for decompression
   }
   ```

**Key Steps:**
```python
# Quantization
y_q_slice = gaussian_conditional. quantize(y_slice, "symbols", mu)

# Entropy encoding with CDF
encoder = BufferedRansEncoder()
encoder.encode_with_indexes(symbols, indexes, cdf, cdf_lengths, offsets)
y_string = encoder.flush()
```

---

#### **3.  decompress(strings, shape) - Bit-stream Decoding**

Reconstructs image from binary strings.

**Process:**
1.  Decompress z from entropy strings → z_hat
2. Generate latent_scales, latent_means from z_hat
3. Iteratively decode y slices using Gaussian conditional
4. Apply synthesis transform → x_hat

**Key Steps:**
```python
# Entropy decoding with CDF
decoder = RansDecoder()
decoder.set_stream(y_string)
rv = decoder.decode_stream(indexes, cdf, cdf_lengths, offsets)

# Dequantization
y_hat_slice = gaussian_conditional.dequantize(rv, mu)
```

---

### Advanced Components

#### **STE (Straight-Through Estimator) Rounding**
```python
def ste_round(x):
    return torch. round(x) - x. detach() + x
    # Forward: round(x)
    # Backward: gradient flows through x directly
```

Allows gradients to flow during training while rounding for compression.

---

#### **Scale Table & Quantiles**

Computes exponential scale table for Gaussian conditional entropy model:

```python
def get_scale_table(min=0.11, max=256, levels=64):
    return torch.exp(
        torch.linspace(log(min), log(max), levels)
    )
    # Creates 64 scale values: [0.11, .. ., 256]
```

---

#### **Slice-Wise Conditional Entropy Coding**

**Why 5 slices?**
- Reduces auto-regressive dependencies (faster encoding)
- Each slice predicted from:
  - Hyper-latent scales/means (z_hat-derived)
  - Previous slices' reconstructions
  - MoE dictionary cross-attention

**Benefits:**
- Parallelize encoding within independent slices
- Adapt to varying image content

---

### State Dict Loading

```python
@classmethod
def from_state_dict(cls, state_dict):
    # Infers N, M from weight shapes if not stored
    N = state_dict["g_a. 0.weight"].size(0)
    M = state_dict["g_a.6.weight"].size(0)
    net = cls(N=N, M=M)
    net.load_state_dict(state_dict)
    return net
```

Handles:
- Automatic architecture discovery from checkpoint
- DataParallel `"module."` prefix handling
- Registered buffer loading (CDF, quantiles)

---

## Architecture Comparison

| Component | ResNet Module | Swin Module | Cross-Attention |
|-----------|---------------|-------------|-----------------|
| **Type** | Convolution-based | Window Attention | Query-Key-Value |
| **Receptive Field** | Local (3×3) | Multi-scale (8×8 window) | Sequence-based (dictionary) |
| **Complexity** | O(C²HW) | O(C²H×W / window²) | O(C × dict_entries) |
| **Best for** | Spatial features | Global context | Entropy modeling |
| **Usage** | Down/upsampling | Feature refinement | Latent conditioning |

---

## Data Flow Summary

```
Image [B, 3, H, W]
    ↓
[g_a: Encode] → y [B, 320, H/16, W/16]
    ↓
[h_a: Hyper-encode] → z [B, 192, H/64, W/64]
    ↓
[Entropy Bottleneck] → z_hat + likelihoods_z
    ↓
[h_z_s: Hyper-decode] → scales, means
    ↓
[Slice Loop with MoE] → y_hat [B, 320, H/16, W/16]
    ↓
[g_s: Decode] → x_hat [B, 3, H, W]
    ↓
Compression Metrics:
  - BPP = -log₂(likelihood) / num_pixels
  - Quality = PSNR, MS-SSIM
```

---

## Novel Contributions Summary

### 🎯 **DictCO = Dictionary Cross-Attention Compression**

1. **MoE Dictionary Cross-Attention** (swin_module.py)
   - K expert dictionaries instead of 1
   - Spatial router for adaptive expert selection
   - Better content-aware entropy modeling

2. **Entropy-Aware Slice Decomposition** (dcae.py)
   - 5 slices with iterative refinement
   - MoE-guided mean/scale prediction
   - Laplace Residual Prediction for fine-tuning

3. **Hybrid Architecture**
   - ResNet for efficient spatial downsampling
   - Swin Transformer for global receptive field
   - Cross-attention for entropy coding

