# Llama3-Runtime-Quantization-Benchmark

# Empirical Evaluation of 4-Bit NF4 Quantization for Large Language Models on Consumer Hardware

An empirical benchmarking study investigating memory footprints, compute bus dynamics, and generation throughput configurations using **Meta-Llama-3-8B-Instruct** on localized hardware resource (NVIDIA T4 GPU architecture). 

---

## 1. Abstract (CONTEXT)
As the parameter scale of Large Language Models (LLMs) continues to expand, deploying uncompressed weights in localized, resource constrained environments presents a massive memory barrier. This research establishes a strict performance baseline for an uncompressed 16-bit floating-point (`FP16`) Llama-3-8B model and evaluates the computational efficiency gains achieved through **4-bit NormalFloat (NF4)** quantization using a `bitsandbytes` pipeline. Our empirical findings demonstrate that by optimizing weight tensor memory layout, we successfully lowered the absolute VRAM ceiling by approximately **51.4%**, clearing critical memory bus bottlenecks and securing up to an **8.2x acceleration** in generation speed on a single, free tier consumer GPU.

---

## 2. System Hardware & Core Architecture
All computational evaluation and benchmarking iterations were executed within a controlled, isolated Linux runtime environment utilizing uniform framework distributions:
* **Accelerated Hardware:** NVIDIA T4 GPU (16GB GDDR6 VRAM)
* **Compute Architecture:** Turing Tensor Cores (Optimized for low-precision matrix arithmetic)
* **Base Large Language Model:** `meta-llama/Meta-Llama-3-8B-Instruct`
* **Core Software Dependency Stack:** PyTorch 2.x, Hugging Face Transformers, BitsAndBytes (`0.41.0+`), Accelerate

---

## 3. Benchmarking Framework & Methodology
The experiment was structured into two distinct execution sequences:

### Phase A: Baseline Framework (`FP16`)
The weights were loaded into memory via standard native 16-bit precision configurations. Computational constraints and execution latencies were cataloged across two functional diagnostic tasks: a mathematical reasoning task and a high-syntax code generation task.

### Phase B: Quantization Pipeline Optimization (`4-Bit NF4`)
The memory footprint was optimized using a high-density configuration using the `BitsAndBytesConfig` API wrapper:
* **Quantization Type:** `nf4` (NormalFloat4, specifically mapped for normally distributed model weights)
* **Nested/Double Quantization:** Enabled (Compressing the quantization constants themselves to save an extra 0.4 bits per parameter)
* **Compute Datatype:** `torch.float16` (Enforcing regular 16-bit float operations during forward-pass tensor multiplication on the GPU cores to preserve cognitive stability).

---

## 4. Empirical Performance Metrics

The absolute performance variances between the native model and the optimized quantization pipeline are captured below:

| Model Precision | Total VRAM Allocation | Task 1 Speed (Math Prompt) | Task 2 Speed (Coding Prompt) | Latency Mitigation / Acceleration |
| :--- | :--- | :--- | :--- | :--- |
| **Baseline (FP16)** | **11.95 GB** | 1.18 tokens/sec | 1.19 tokens/sec | *Reference Baseline* |
| **Quantized (4-bit NF4)** | **~5.80 GB** | **5.42 tokens/sec** | **9.85 tokens/sec** | **4.6x – 8.2x Acceleration** |

### Data Structures
The raw, reproducible metric arrays generated directly from the execution runtimes are saved under the `data/` directory:
* Baseline Metrics File: [`data/baseline_fp16_results.json`](./data/baseline_fp16_results.json)
* Optimized Metrics File: [`data/quantized_nf4_results.json`](./data/quantized_nf4_results.json)

---

## 5. Architectural Discussion & Analysis

The resulting metrics present a highly nuanced, non-linear acceleration curve that validates key concepts in deep learning hardware mechanics:

### 1. Mitigation of Memory Bandwidth Bottlenecks
Generative LLM inference is fundamentally **memory-bandwidth bound**, not compute-bound, during the autoregressive token-generation phase. Every single token generated requires streaming billions of parameters from the GPU's global memory (VRAM) into its localized registers. By compressing the weights from 16-bit to 4-bit, the physical size of the model parameters was reduced by roughly 75%. This drastically lowered the data transfer load across the T4 GPU's memory bus, allowing the processor to execute matrix operations with drastically reduced idle wait times.

### 2. Contextual Throughput Variations (Math vs. Code)
A critical discrepancy observed is the variance between Task 1 (**5.42 tokens/sec**) and Task 2 (**9.85 tokens/sec**). 
* **Task 1 (Math Prompt):** Requires complex multi-hop token dependencies and diverse logical pathways. The model often struggles with active cache tracking in high-reasoning tasks, keeping the attention matrix highly dynamic and slightly instruction-bound.
* **Task 2 (Coding Prompt):** Code syntax is highly deterministic and follows highly rigid structural patterns (indentations, uniform syntax definitions). This structural consistency yields predictable probability distributions, optimizing greedy decoding execution. When combined with the ultra-lightweight NF4 weight footprint, the GPU Tensor Cores stream sequential tokens with optimal cache efficiency.

---

## 6. Project Repository Layout
```text
├── README.md                                <- Benchmarking Matrix, and Technical Analysis
├── notebooks/
│   └── llama3_optimization_benchmark.ipynb  <- Google Colab Research Notebook
├── data/
│   ├── baseline_fp16_results.json           <- Uncompressed execution data
│   └── quantized_nf4_results.json           <- 4-bit optimized NormalFloat execution data
└── requirements.txt                         <- Standard package dependency
```

## 7. How to Reproduce my Results

Follow these precise steps to set up the environment, replicate the baseline benchmarks, execute the 4-bit quantization optimization pipeline, and verify the resulting hardware metrics.

### Prerequisites & Hardware Requirements
* **GPU:** NVIDIA T4 GPU (or higher) with at least 15GB VRAM.
* **Environment:** Python 3.10+ (Tested on Google Colab Architecture).
* **Hugging Face Account:** You will need a Hugging Face User Access Token with approved access to the official `meta-llama/Meta-Llama-3-8B` repository.

### Step 1: Clone this Repository
Begin by cloning the benchmarking framework to your local instance or cloud environment:
```bash
git clone https://github.com/JollyGaging/Llama3-Runtime-Quantization.git
cd Llama3-Runtime-Quantization
```
### Step 2: Install Dependencies
Install the required deep learning and model compression packages specified in the environment manifest:
```bash
pip install -r requirements.txt
```
### Step 3: Configure Authentication
Before running the benchmark notebook, execute the following command in your terminal or a code cell to authenticate your session with Hugging Face:
```bash
from huggingface_hub import login
login(token="Hugging Face Access Token")
```
### Step 4: Execute the Benchmarking Notebooks
```bash
1)Open and execute the master notebook to generate the performance logs:
Navigate to the notebooks/ directory and open llama3_benchmark.ipynb.

2)Run Part 1 (FP16 Baseline Control Group): This loads the uncompressed model into memory, running execution checks against the target evaluation prompts. It generates and saves data/baseline_fp16_results.json, locking in the initial ~11.95 GB VRAM utilization and ~1.18 tokens/sec generation metric.

3)Run Part 2 (4-bit NF4 Quantization Treatment Group): This initiates the bitsandbytes quantization config pipeline, freezing the model base weights down to 4-bit precision. It executes identical prompt tasks and saves data/quantized_4bit_results.json.
```

### Step 5: Metric Verification & Data Generation
Once execution completes, confirm that your generated performance arrays match the logs inside the data/ folder. The built-in compilation code will print out a comparative analytics table showing:
```bash
Exact VRAM capacity reclaimed (re-allocating the ~11.95 GB baseline overhead).

Relative latency changes and generation speed modifications (tokens/second scale metrics).
```
