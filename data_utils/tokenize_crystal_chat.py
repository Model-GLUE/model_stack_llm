from functools import partial
import os
import fire
import glob
import json
import multiprocessing
import tqdm
from transformers import AutoTokenizer


CRYSTAL_CHAT_DIR = './crystal_chat_raw'
OUTPUT_DIR = './data_tokenized'
WORDS_TO_TOK = 8192
MULTI_PROCESSING_CHUNK_SIZE = 500
HF_CACHE_DIR = './hf_cache'
HF_TOKEN = None


def tokenize(text, tokenizer, prefix_ids=None, suffix_ids=None):
    words = text.split(' ')

    token_ids = []
    for i in range(0, len(words), WORDS_TO_TOK):
        token_ids.extend(tokenizer(
            ' '.join(words[i:i + WORDS_TO_TOK]),
            add_special_tokens=False
        )['input_ids'])

    if prefix_ids is not None:
        token_ids = prefix_ids + token_ids
    if suffix_ids is not None:
        token_ids = token_ids + suffix_ids

    return token_ids


def tokenize_example(line, data_type, tokenizer):
    example = json.loads(line)
    if data_type == 'chat':
        token_ids, tgt_mask = [], []
        for conv in example['conversations']:
            if conv['from'] in ['human', 'system']:
                prefix_ids = None
                suffix_ids = None
                is_tgt = 0
            else:
                assert conv['from'] == 'gpt'
                prefix_ids = None
                suffix_ids = [tokenizer.eos_token_id]
                is_tgt = 1

            uttr_token_ids = tokenize(
                text=conv['value'],
                tokenizer=tokenizer,
                prefix_ids=prefix_ids,
                suffix_ids=suffix_ids)
            token_ids = token_ids + uttr_token_ids
            tgt_mask = tgt_mask + [is_tgt] * len(uttr_token_ids)

    elif data_type == 'text':
        token_ids = tokenize(
            text=example['markdown'],
            tokenizer=tokenizer,
            suffix_ids=[tokenizer.eos_token_id])
        tgt_mask = [1] * len(token_ids)

    return json.dumps({'token_ids': token_ids, 'tgt_mask': tgt_mask})


def get_n_lines(filename):
    n_lines = 0
    for _ in open(filename):
        n_lines += 1
    return n_lines


def main(model_name_or_path='modelglue-stacking/llama_7b_chat_code_stacking', n_procs=None):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path, token=HF_TOKEN)

    files = {'chat': [], 'text': []}
    for filename in glob.glob(f'{CRYSTAL_CHAT_DIR}/*.jsonl'):
        if (filename.split('/')[-1] not in
                ['textbooks.jsonl', 'program_books.jsonl']):
            files['chat'].append(filename)
        else:
            files['text'].append(filename)

    for data_type in files:
        tokenize_example_fn = partial(
            tokenize_example, data_type=data_type, tokenizer=tokenizer)

        os.makedirs(f'{OUTPUT_DIR}/{data_type}', exist_ok=True)
        for filename in sorted(files[data_type]):
            output_filename = (
                    f'{OUTPUT_DIR}/{data_type}/' + filename.split('/')[-1])

            with open(output_filename, 'w') as output_file:
                n_process = n_procs if n_procs is not None else os.cpu_count()
                with multiprocessing.Pool(processes=n_process) as pool:
                    for json_str in pool.imap_unordered(
                            tokenize_example_fn, tqdm.tqdm(
                                open(filename), desc=f'Tokenizing {filename}'
                            ), chunksize=MULTI_PROCESSING_CHUNK_SIZE):
                        print(json_str, file=output_file, flush=True)


if __name__ == '__main__':
    fire.Fire(main)
