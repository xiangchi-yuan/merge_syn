# Model Merge Can Improve Reasoning Data Synthesize

## Background

Recent research reveals that **iteratively fine-tuning (FT) large language models (LLMs) with synthetic data** can iteratively improve model performance. After each fine-tuning step, the model becomes stronger and synthesizes better data. However, this iterative process has significant limitations:
- After 1-2 iterations, the model's performance plateaus or even degrades.
- **Reason**: While data quality increases after each iteration, the diversity of the data decreases, limiting further improvements.

## Our Solution: Model Merge

To address this, we propose a **model merging approach**:
- We merge the original model weights with those of previous models from the self-improvement iterations.
- This process balances synthetic data quality and diversity.
- Additionally, merging weights introduces **extra model generalization**, further enhancing performance.

## How to Run the Code

### Step 1: Generate Data
```bash
python generate_gms8k.py
```

### Step 2: Fine-Tune the Model
```bash
bash train_qwen.sh
```

### Step 3: Merge Models
```bash
python MergeLM-main/merge_llms_instruct_math_code.py \
    --merge_instruct \
    --merge_math \
    --merging_method_name mask_merging \
    --use_weight_rescale \
    --weight_mask_rate 0.2 \
    --mask_apply_method average_merging \
    --tensor_parallel_size 1
```

### Key Arguments for Merging Script:
- `--merge_instruct`: Merge using instruction-tuned data.
- `--merge_math`: Merge using math-tuned data.
- `--merging_method_name`: Specify the merging method (e.g., `mask_merging`).
- `--use_weight_rescale`: Enable weight rescaling during merging.
- `--weight_mask_rate`: Set the mask rate for merging (default: `0.2`).
- `--mask_apply_method`: Apply the mask using the specified method (e.g., `average_merging`).
- `--tensor_parallel_size`: Set the tensor parallel size (default: `1`).

## Summary

This repository provides an effective approach to improve LLM reasoning capabilities by combining **synthetic data generation, fine-tuning, and weight merging**. The proposed **model merging method** ensures a balance between data quality and diversity while enhancing the generalization capability of the model.

