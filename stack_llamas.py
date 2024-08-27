
import argparse
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
import torch
import torch.nn.functional as F
from collections import OrderedDict
from itertools import chain

def padded_sum(w1, w2):
    if w1 is None:
        return w2
    elif w2 is None:
        return w1
    else:
        assert w1.ndim == w2.ndim
        pad1 = list(chain.from_iterable([[0, max(0, w2.size(i) - w1.size(i))] for i in range(w1.ndim)][::-1]))
        pad2 = list(chain.from_iterable([[0, max(0, w1.size(i) - w2.size(i))] for i in range(w1.ndim)][::-1]))

        w1 = F.pad(w1, pad1, mode='constant', value=0.)
        w2 = F.pad(w2, pad2, mode='constant', value=0.)

        return w1 + w2
    
def is_intermediate_layer(k):
    return k.startswith('model.layers.')
 
def get_num_layers(state_dict):
    num_layers = 0
    for k in state_dict.keys():
        if not is_intermediate_layer(k):
            continue
        num_layers = max(num_layers, int(k.split('.')[2]))
    return num_layers + 1

def stack_layers(model_ckpts, output):
    current_layer = 0
    for ckpt in model_ckpts:
        num_layers = 0
        for k, v in ckpt.items():
            if not is_intermediate_layer(k):
                continue
            fields = k.split('.')
            l = int(fields[2])
            num_layers = max(num_layers, l + 1)
            fields = fields[:2] + [str(l + current_layer)] + fields[3:]
            output['.'.join(fields)] = v
        current_layer += num_layers
        
    return output

def interleave_layers(model_ckpts, output):
    parse_layer_num = lambda k: int(k.split('.')[2])
    model_layers = []
    for i, ckpt in enumerate(model_ckpts):
        model_layers

    current_layer = 0
    inner_model_layer = 0
    finished = False
    while not finished:
        finished = True
        for i, ckpt in enumerate(model_ckpts):
            for k, v in ckpt.items():
                if not is_intermediate_layer(k):
                    continue
                fields = k.split('.')
                l = int(fields[2])
                if l == inner_model_layer:
                    fields = fields[:2] + [str(current_layer)] + fields[3:]
                    output['.'.join(fields)] = v
                    finished = False
            current_layer += 1
        inner_model_layer += 1
        
    return output
        
def merge_non_layers(model_ckpts, k, output):

    # Merge embedding layers
    param = None
    ncnt = None
    for ckpt in model_ckpts:
        param = padded_sum(param, ckpt[k])
        ncnt = padded_sum(ncnt, torch.ones_like(ckpt[k]))
    output[k] = param / ncnt
    
    return output

def main():
    # Set up the argument parser
    parser = argparse.ArgumentParser(description="Receive deepen model's args")
    parser.add_argument("--model_path", nargs='*', required=True, type=str, help="original model path")
    parser.add_argument("--tokenizer_name", default=None, type=str, help="original model path")
    parser.add_argument("--stack_method", default='stack', type=str, help="stacking methods: stack/interleave")
    parser.add_argument("--reset_wo", default=False, action='store_true', help="Reset W_o and W_v layers similar to Llama-Pro.")
    parser.add_argument("--output_path", default=None, type=str, help="path to save model ckpt")
    parser.add_argument("--push_to_hub", default=None, type=str, help="remote hub to save model")

    # Parse the arguments
    args = parser.parse_args()
    
    output = OrderedDict()
    
    ckpt_states = []
    nonlayer_keys = None
    for path in args.model_path:
        model = AutoModelForCausalLM.from_pretrained(path,
            torch_dtype=torch.float16,
            # low_cpu_mem_usage=True,
            # trust_remote_code=trust_remote_code,
            use_safetensors=False
        )
        state_dict = model.state_dict()
        ckpt_states.append(state_dict)
        del model
        
        current_nonlayer_keys = [k for k in state_dict.keys() if not is_intermediate_layer(k)]
        if nonlayer_keys is None:
            nonlayer_keys = current_nonlayer_keys
        else:
            # sanity check
            if nonlayer_keys != current_nonlayer_keys:
                print("Architectures are not compatible!", current_nonlayer_keys - nonlayer_keys)
    
    if args.stack_method == 'stack':
        stack_layers(ckpt_states, output)
    elif args.stack_method == 'interleave':
        interleave_layers(ckpt_states, output)
    else:
        raise NotImplementedError

    for k in nonlayer_keys:
        merge_non_layers(ckpt_states, k, output)
        
    print(output.keys())

    config_update = dict(
        num_hidden_layers = get_num_layers(output),
        vocab_size = output['model.embed_tokens.weight'].shape[0]
    )
    config = AutoConfig.from_pretrained(args.model_path[0], **config_update)
    model = AutoModelForCausalLM.from_config(config)
    model.load_state_dict(output)
    if args.output_path is not None:
        model.save_pretrained(args.output_path)
    if args.push_to_hub is not None:
        model.push_to_hub(args.push_to_hub, private=True)

    tokenizer_name = args.tokenizer_name if args.tokenizer_name is not None else args.model_path[0]
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True, trust_remote_code=True)
    if args.output_path is not None:
        tokenizer.save_pretrained(args.output_path)
    if args.push_to_hub is not None:
        tokenizer.push_to_hub(args.push_to_hub, private=True)

if __name__ == "__main__":
    main()
