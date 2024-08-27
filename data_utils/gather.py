from functools import partial
import os
import fire
import glob
import json
import tqdm
import multiprocessing
import datasets

TOKENIZED_DIR = './data_tokenized'
OUTPUT_DIR = './crystal_chat_raw'
FINAL_DIR = './data_final'
CHUNK_SIZE = 50000
SEED = 11111


def gather_file(filename, config_str, context_length):
    data_type = filename.split('/')[-2]
    subset_name = filename.split('/')[-1]

    os.makedirs(f'{OUTPUT_DIR}/{config_str}/{data_type}', exist_ok=True)
    with (open(f'{OUTPUT_DIR}/{config_str}/{data_type}/{subset_name}', 'w')) as output_file:
        token_ids_buffer, tgt_mask_buffer = [], []
        for line in tqdm.tqdm(open(filename), desc=filename):
            example = json.loads(line)
            token_ids_buffer.extend(example['token_ids'])
            tgt_mask_buffer.extend(example['tgt_mask'])

            while len(token_ids_buffer) >= context_length:
                print(json.dumps({
                    'token_ids': token_ids_buffer[:context_length],
                    'tgt_mask': tgt_mask_buffer[:context_length],
                    'data_type': data_type,
                    'subset_name': subset_name
                }), file=output_file, flush=True)
                token_ids_buffer = token_ids_buffer[context_length:]
                tgt_mask_buffer = tgt_mask_buffer[context_length:]

        assert len(token_ids_buffer) == len(tgt_mask_buffer)


def main(context_length=8192, n_epochs=3):
    valid_filenames = []
    for filename in sorted(glob.glob(f'{TOKENIZED_DIR}/*/*.jsonl')):
        data_type = filename.split('/')[-2]
        assert data_type in ['chat', 'text']

        valid_filenames.append(filename)
    print(valid_filenames)

    gather_fn = partial(gather_file, config_str=f'cxt_len_{context_length}', context_length=context_length)
    with multiprocessing.Pool(processes=os.cpu_count()) as pool:
        pool.map(gather_fn, valid_filenames)

    print(sorted(glob.glob(f'{OUTPUT_DIR}/*/*.jsonl')))

    dataset = datasets.load_dataset(
        'json',
        data_files=sorted(glob.glob(f'{OUTPUT_DIR}/*/*/*.jsonl')),
        split='train',
        cache_dir='./cache',
        num_proc=os.cpu_count())

    n_chunks = len(dataset) // CHUNK_SIZE + 1
    for epoch_idx in range(n_epochs):
        dataset = dataset.shuffle(seed=SEED)

        os.makedirs(f'{FINAL_DIR}/', exist_ok=True)
        for chunk_idx in range(n_chunks):
            output_filename = f'{FINAL_DIR}/chunk-{chunk_idx}.jsonl'
            chunk = dataset.shard(num_shards=n_chunks, index=chunk_idx)
            print(f'Saving {output_filename} ...')
            chunk.to_json(output_filename, num_proc=os.cpu_count())


if __name__ == '__main__':
    fire.Fire(main)