```mermaid
graph TD
    Input(Input Context) --> Linear[Linear Proj]
    Linear --> DWT[Learnable DWT]
    
    subgraph "Frequency Domain Processing"
    DWT -- "Low Freq (LL)" --> NormL[LayerNorm]
    NormL --> DictAttn[Dictionary Attention Global Coarse Info]
    
    DWT -- "High Freq (HF)" --> Router{Router}
    NormL -.-> Router
    
    Router -- "Logits + Bias" --> TopK[Top-K Selection]
    TopK --> Experts[Sparse Experts Fine Texture Info]
    
    DictAttn --> ProcessedLL
    Experts --> ProcessedHF
    end
    
    ProcessedLL & ProcessedHF --> IDWT[Inverse DWT]
    IDWT --> MSA[Multi-Scale Aggregation]
    MSA --> MLP[ConvFFN]
    MLP --> Output
    
    Input -.->|Residual| Output

    style DWT fill:#ff9,stroke:#333
    style IDWT fill:#ff9,stroke:#333
    style Experts fill:#9f9,stroke:#333
    style Router fill:#f99,stroke:#333
```


To provide the most detailed view for your paper, here are the **Class Diagrams broken down by specific module**. You can generate these using Mermaid.

These diagrams focus on **attributes (variables)** and **methods (functions)** to show exactly how data is stored and processed.

### 1. The Main DCAE Model (Architecture)
This diagram shows the high-level container that manages the compression pipeline, the entropy bottlenecks, and the slicing groups.

```mermaid
classDiagram
    note "File: models/dcae.py"
    class DCAE {
        %% Configuration
        +int N = 192
        +int M = 320
        +List[int] groups
        +int num_slices

        %% Sub-Modules
        +nn.Sequential g_a (Analysis Encoder)
        +nn.Sequential g_s (Synthesis Decoder)
        +nn.Sequential h_a (Hyper Encoder)
        +nn.Sequential h_z_s1 (Hyper Scale)
        +nn.Sequential h_z_s2 (Hyper Mean)
        
        %% Entropy Modules
        +EntropyBottleneck entropy_bottleneck
        +GaussianConditional gaussian_conditional
        +ModuleList dt_cross_attention (SpectralMoE)
        +ModuleList context_transforms (NAFBlocks)

        %% Methods
        +forward(x) Dict
        +compress(x) Dict
        +decompress(strings, shape) Tensor
        +update(scale_table)
    }

    class NAFBlock {
        +Conv2d dwconv
        +Sequential sca (Attention)
        +Sequential FFN
        +Parameter beta
        +Parameter gamma
        +forward(x)
    }

    DCAE *-- NAFBlock : "Uses for Context"
    DCAE ..> GaussianConditional : "Predicts Params"
```

---

### 2. The Spectral MoE Module (The Core Innovation)
This is the most important diagram for your methodology section. It details the **Wavelet Transform** and the **Mixture of Experts** internals, including the routing cache.

```mermaid
classDiagram
    note "File: modules/new_swin_module.py"
    
    class SpectralMoEDictionaryCrossAttention {
        %% Dimensions
        +int input_dim
        +int dim_low
        +int dim_high
        +int num_experts
        
        %% Wavelets
        +LearnableWaveletTransform dwt
        +InverseLearnableWaveletTransform idwt
        
        %% Low Frequency Path
        +Parameter dict_low
        +Linear q_low, k_low
        
        %% High Frequency MoE
        +Sequential router
        +Parameter experts_high
        +Parameter expert_biases (Loss-Free Buffer)
        +Linear q_high, k_high, v_all

        %% Cache for Loss
        +Tensor last_routing_logits
        +Tensor last_routing_indices

        %% Methods
        +forward(x)
        +process_low_freq(x)
        +process_high_freq_guided(hf, lf)
    }

    class LearnableWaveletTransform {
        +LiftingBlock P_horz
        +LiftingBlock U_horz
        +LiftingBlock P_vert
        +LiftingBlock U_vert
        +forward(x) Tuple[ll, hf]
    }

    class LiftingBlock {
        +Sequential net (Conv+GELU)
        +forward(x)
    }

    SpectralMoEDictionaryCrossAttention *-- LearnableWaveletTransform
    LearnableWaveletTransform *-- LiftingBlock
```

---

### 3. The Backbone Blocks (Swin & ResNet)
This diagram explains the building blocks of your Encoder (`g_a`) and Decoder (`g_s`).

```mermaid
classDiagram
    note "File: modules/new_swin_module.py"
    
    class SwinBlockWithConvMulti {
        +ModuleList layers
        +Conv2d conv
        +forward(x)
    }

    class ResScaleConvGateBlock {
        +LayerNorm ln1
        +WMSA msa (Window Attention)
        +LayerNorm ln2
        +ConvGELU mlp
        +Scale res_scale_1
        +Scale res_scale_2
        +forward(x)
    }

    class WMSA {
        +int window_size
        +Parameter relative_position_params
        +Linear embedding_layer
        +forward(x)
    }

    class ConvGELU {
        +Linear fc1
        +DWConv dwconv
        +Linear fc2
        +forward(x)
    }

    SwinBlockWithConvMulti *-- ResScaleConvGateBlock
    ResScaleConvGateBlock *-- WMSA
    ResScaleConvGateBlock *-- ConvGELU
```

---

### 4. The Loss & Training System
This diagram illustrates how the **Loss Function** calculates gradients and how the **LossFreeBalancer** interacts with the model from the outside.

```mermaid
classDiagram
    note "File: loss/new_loss.py & train.py"

    class RateDistortionLoss {
        +float lmbda
        +float alpha_moe
        +bool use_loss_free_balancing
        +MSELoss mse
        +CharbonnierLoss charbonnier
        +forward(output, target)
        -load_balancing_loss_func(logits)
    }

    class LossFreeBalancer {
        +int num_experts
        +float update_rate
        +update_model_biases(model, router_data)
    }

    class Trainer {
        +train_one_epoch()
        +main()
    }

    Trainer --> RateDistortionLoss : "1. Calculates Gradients"
    Trainer --> LossFreeBalancer : "2. Updates Bias Buffer"
    LossFreeBalancer ..> SpectralMoEDictionaryCrossAttention : "Modifies expert_biases"
```

