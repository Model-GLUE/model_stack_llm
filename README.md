# Model-GLUE : Democratized LLM Scaling for A Large Model Zoo in the Wild

This is the official code for the **stacking** benchmark in the ``Model-GLUE : Democratized LLM Scaling for A Large Model Zoo in the Wild``.


## Getting Started

### Installation

To run our code, we require the following package to be installed.
```
torch>=2.1.1
lightning>=2.1.2
transformers>=4.36.2
datasets
fire
wandb
gdown==5.1.0
lm_eval[vllm]==0.4.0
```

### Data preperation

In our experiments, model stacking needs continual training on an instruction fine-tuning dataset. We provide the following handy scripts to download and preprocess [CrystalChat](https://huggingface.co/LLM360/CrystalChat) dataset.

```
# Downloading
bash data_utils/download_crystal_chat_raw.sh

# Tokenization
python data_utils/tokenize_crystal_chat.py --model_name_or_path=<tokenizer_path>

# Chunking and Gathering
python data_utils/gather.py --context_length 1024 --n_epochs 1
```

## Experiments

### Stacking

To stack a series of models, run the following command:

```
python stack_llamas.py --model_path <model_1> <model_2> ... --stack_method <stack | interleave>  --output_path <output_dir> --push_to_hub <hf_repo_id>
```

where `--model_path` can be any LLaMA family models on HuggingFace.

### Training

After obtainining a checkpoint of stacked models, one can train it via the following command:
```
python main.py \
    --hf_model_name_or_path <huggingface_model_path> \
    --work_dir <working_dir> \
    --n_nodes <number_of_nodes> \
    --n_devices_per_node <gpus_per_node> \
    --n_epochs 10 \
    --data_file data.jsonl \
    --per_device_batch_size 10 \
    --accumulate_grad_batches 1
```

### Evaluation

The fine-tuned model can be evaluated through the scripts below:

```
python eval_utils/main.py <ckpt_dir>
```

And the full results can be quickly viewd by the following script:

```
python present_results.py <ckpt_dir>
```

## Acknowledgement

Part of the codes are adapted from [LiGO](https://github.com/VITA-Group/LiGO), [LLM360](https://www.llm360.ai/) and [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness).