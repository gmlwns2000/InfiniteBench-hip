import json
import os
from pathlib import Path
import time
from typing import List, Tuple, Any

import torch
from torch import Tensor
from transformers import AutoTokenizer
from transformers.modeling_outputs import BaseModelOutputWithPast

from eval_utils import (
    dump_jsonl,
    create_prompt,
    load_data,
    get_answer,
    DATA_NAME_TO_MAX_NEW_TOKENS,
)
from args import parse_args


IS_DEEPSEEK = os.getenv('IS_DEEPSEEK', '0') == '1'
USING_SGLANG = os.getenv('USING_SGLANG', '0') == '1'
SGLANG_PORT = int(os.getenv('SGLANG_PORT', '30000'))
MAX_POSITION_ID = int(os.getenv('SEQ_LEN', '128')) * 1024  # Determined by the model
TRUNCATE_LEN = int(os.getenv('SEQ_LEN', '128')) * 1024


def truncate_input(input: list, max_length: int, manner="middle"):
    if len(input) <= max_length:
        return input
    if manner == "middle":
        split = max_length // 2
        return input[0:split] + input[-split:]
    else:
        return None


def truncate_by_tokens(input, tok, max_tokens, manner: str = "middle", quite = False):
    tokens = tok.encode(input, add_special_tokens=False)
    len_before = len(tokens)
    if not quite:
        print(f"# tokens before: {len_before}, ", end='')
    tokens = truncate_input(tokens, max_length=max_tokens, manner=manner)
    tokens = tok.encode(tok.decode(tokens, skip_special_tokens=False), add_special_tokens=False)
    len_after = len(tokens)  # type: ignore
    if not quite:
        print(f"# tokens after: {len_after}")
    # print(tokens[:20], tokens[-20:])
    assert len_after <= (len_before + 16)
    assert len_after <= (max_tokens + 16)
    return tok.decode(tokens, skip_special_tokens=False), len_after


def chunk_generate(
    model,
    tok,
    texts: List[str],
    max_tokens: int,
    sliding_window: int = MAX_POSITION_ID,
    chunk_size: int = 2500,
    verbose: bool = False,
) -> List[str]:
    """
    Directly performing inference using HF transformers will result in OOM
    when using one A100 GPU. This is because the attention matrix is too large,
    so we chunk the input up and perform forward pass on each chunk to build
    up the KV cache. Note that each token still has to attend to
    all tokens in the past.
    """
    with torch.no_grad():
        """
        input_ids: (b, n)
        attention_mask: (b, n)
        [
            [0, 0, .., 0, 1, 1, ..., 1]
            ...
        ]
        """
        inputs = tok(texts, return_tensors="pt", padding=False, add_special_tokens=False)
        if model is None:
            inputs = inputs.to('cpu')
        else:
            inputs = inputs.to(model.device)  # type: ignore
        input_ids: Tensor = inputs.input_ids  # (b, n)
        
        # attention_mask: Tensor = inputs.attention_mask  # (b, n)
        # position_ids: Tensor = attention_mask.long().cumsum(dim=-1) - 1
        # position_ids.masked_fill_(attention_mask == 0, value=1)
        # seq_len = input_ids.shape[-1]
        # print("seq_len:", seq_len)
        # kv_cache: Any = None
        # # Split into chunks for pre-filling
        # chunk_idxs = []
        # n = seq_len - 1
        # while n > 0:
        #     chunk_idxs.append(n)
        #     n -= chunk_size
        # chunk_idxs.append(0)
        # chunk_idxs = chunk_idxs[::-1]
        # chunk_lo = chunk_idxs[:-1]
        # chunk_hi = chunk_idxs[1:]
        # print(f"Number of chunks: {len(chunk_lo)}, generating...")
        # start_time = time.time()
        # for chunk_i, (chunk_lo, chunk_hi) in enumerate(
        #     zip(chunk_lo, chunk_hi)
        # ):
        #     if verbose:
        #         print(
        #             f"[chunk {chunk_i}] {chunk_lo} : {chunk_hi}",
        #             round(time.time() - start_time),
        #         )
        #     chunk_input_ids = input_ids[:, chunk_lo:chunk_hi]
        #     if kv_cache is not None:
        #         mask_start_idx = chunk_lo - kv_cache[0][0].shape[2]
        #     else:
        #         mask_start_idx = chunk_lo
        #     chunk_attention_mask = attention_mask[:, mask_start_idx:chunk_hi]
        #     chunk_position_ids = position_ids[:, chunk_lo:chunk_hi]
        #     outputs: BaseModelOutputWithPast = model.model.forward(
        #         input_ids=chunk_input_ids,
        #         attention_mask=chunk_attention_mask,
        #         position_ids=chunk_position_ids,
        #         past_key_values=kv_cache,
        #         return_dict=True,
        #         use_cache=True,
        #     )
        #     kv_cache = outputs.past_key_values
        #     # Discard KV states on the left beyond the window
        #     new_cache = ()
        #     n_layers = len(kv_cache)
        #     for layer_i in range(n_layers):
        #         keys = kv_cache[layer_i][0][:, :, -sliding_window:]
        #         values = kv_cache[layer_i][1][:, :, -sliding_window:]
        #         new_cache += ((keys, values),)
        #     kv_cache = new_cache
        # kv_cache_len = kv_cache[0][0].shape[2]
        # outputs = model.generate(
        #     input_ids=input_ids[:, :],
        #     attention_mask=attention_mask[:, -kv_cache_len - 1 :],
        #     max_new_tokens=max_tokens,
        #     past_key_values=kv_cache,
        #     eos_token_id=tok.pad_token_id,
        #     use_cache=True,
        #     do_sample=False,
        # )
        
        # print(tok.decode(input_ids[0], skip_special_tokens=False)[:500], tok.decode(input_ids[0], skip_special_tokens=False)[-500:])
        
        if USING_SGLANG:
            import requests

            prompt_text = tok.decode(input_ids[0], skip_special_tokens=False)
            
            if IS_DEEPSEEK:
                sampling_params = {
                    "top_p": 1.0,
                    "temperature": 0.4,
                    "max_new_tokens": 4096,
                }
            else:
                sampling_params = {
                    # "top_k": 1, # greedy
                    "top_p": 1e-6,
                    "max_new_tokens": max_tokens,
                }
            
            response = requests.post(
                f"http://localhost:{SGLANG_PORT}/generate",
                json={
                    "text": prompt_text,
                    "sampling_params": sampling_params,
                },
            )
            assert response.status_code == 200, response.json()
            # print(response.json())
            
            decoded = response.json()['text'] # type: str
            
            if IS_DEEPSEEK:
                # print(f'RAW decoded: \n--------------\n{decoded}\n--------------\n')
                # print('</think>' in decoded, decoded.find('</think>'))
                if '</think>' in decoded:
                    start_idx = decoded.find('</think>')
                    decoded = decoded[start_idx + len('</think>'):].strip()
                else:
                    decoded = decoded[-max_tokens*3:]
            
            responses = [decoded]
        else:
            outputs = model.generate(
                input_ids=input_ids,
                max_new_tokens=max_tokens,
                # eos_token_id=tok.pad_token_id,
                do_sample=False,
            )
        
            responses = [
                tok.decode(t[input_ids.shape[-1]:], skip_special_tokens=True) for t in outputs
            ]
    return responses


def get_pred(
    model,
    tok: AutoTokenizer,
    input_text: str,
    max_tokens: int,
    verbose: bool = False,
    reqiure_truncate: bool = True,
) -> str:
    """
    Truncate down to 128k then make inference.
    """
    if reqiure_truncate:
        print("Truncating... ", end = '')
        input_text, len_after = truncate_by_tokens(input_text, tok, TRUNCATE_LEN - max_tokens - 32)
    else:
        len_after = 0
    if verbose:
        print("# chars:", len(input_text))
        print("=============== Input ===============")
        print(input_text[:200])
        print("...")
        print(input_text[-200:])
        print("=====================================")
    output = chunk_generate(
        model,
        tok,
        [input_text],
        max_tokens=max_tokens,
        chunk_size=4096,
        verbose=verbose,
    )[0]
    output = output.replace('<|eot_id|>', '')
    output = output.replace('<eos>', '')
    output = output.replace('<end_of_turn>', '')
    output = output.replace('[|endofturn|]', '')
    print("Chunked generation:", output.replace('\n', '\\n'))
    return output, len_after


from hip.models.modeling_llama import LlamaForCausalLM
from transformers import LlamaForCausalLM as OriginalLlamaForCausalLM

ATTENTION_METHOD = os.getenv('ATTENTION_METHOD', 'hip')

def load_model(
    model_name: str,
) -> Tuple["LlamaForCausalLM", AutoTokenizer]:
    
    print("Loading tokenizer")
    tok = AutoTokenizer.from_pretrained(model_name)
    # tok.pad_token = tok.eos_token
    print("Loading model")
    start_time = time.time()
    if not USING_SGLANG:
        model = LlamaForCausalLM.from_pretrained(
            model_name, 
            device_map="auto", 
            torch_dtype=torch.bfloat16,
        )
    
        for m in model.modules():
            if hasattr(m, 'attention_method'):
                m.attention_method = ATTENTION_METHOD
    else:
        model = None
    
    print("Time taken:", round(time.time() - start_time))
    return model, tok  # type: ignore


if __name__ == "__main__":
    args = parse_args()
    IS_EXAONE = os.getenv('IS_EXAONE', '0') == '1'
    IS_GEMMA = os.getenv('IS_GEMMA', '0') == '1'
    IS_MISTRAL = os.getenv('IS_MISTRAL', '0') == '1'
    if IS_EXAONE:
        model_name = f"exaone3-{TRUNCATE_LEN // 1024}-{args.model_name}"
    elif IS_GEMMA:
        model_name = f'gemma2-{TRUNCATE_LEN // 1024}-{args.model_name}'
    elif IS_MISTRAL:
        model_name = f'mistral-{TRUNCATE_LEN // 1024}-{args.model_name}'
    else:
        model_name = f"llama3-{TRUNCATE_LEN // 1024}-{args.model_name}"

    print(json.dumps(vars(args), indent=4))
    data_name = args.task

    # Model
    max_tokens = DATA_NAME_TO_MAX_NEW_TOKENS[data_name]
    model, tok = load_model(args.model_path)

    # Data
    result_dir = Path(args.output_dir, model_name)
    result_dir.mkdir(exist_ok=True, parents=True)
    examples = load_data(data_name, data_dir=args.data_dir)

    if args.stop_idx is None:
        args.stop_idx = len(examples)
        output_path = (
            result_dir / f"preds_{data_name}.jsonl"
        )
    else:
        output_path = (
            result_dir / f"preds_{data_name}_{args.start_idx}-{args.stop_idx}.jsonl"  # noqa
        )

    preds = []
    print("==== Evaluation ====")
    print(f"# examples: {len(examples)}")
    print(f"Start index: {args.start_idx}")
    print(f"Stop index: {args.stop_idx}")
    print(f"Verbose: {args.verbose}")
    print(f"Max tokens: {max_tokens}")
    input_texts = []
    import threading
    for i in range(args.start_idx):
        input_texts.append(None)
    
    def thread_main():
        for i in range(args.start_idx, args.stop_idx):
            eg = examples[i]
            input_text = create_prompt(eg, data_name, model_name, args.data_dir)
            # print("Truncating... ", end = '')
            input_text, _ = truncate_by_tokens(input_text, tok, TRUNCATE_LEN - max_tokens - 32, quite=True)
            input_texts.append(input_text)
    t = threading.Thread(target=thread_main, daemon=True)
    t.start()
    
    for i in range(args.start_idx, args.stop_idx):
        eg = examples[i]
        print(f"====== Example {i} ======")
        while len(input_texts) <= i:
            time.sleep(0.001)
        input_text = input_texts[i]
        pred, len_after = get_pred(
            model, tok, input_text,
            max_tokens=max_tokens,
            verbose=args.verbose,
            reqiure_truncate=False,
        )
        if args.verbose:
            print(pred)
        preds.append(
            {
                "id": i,
                "prediction": pred,
                "ground_truth": get_answer(eg, data_name),
                "context_length": len_after,
            }
        )
        dump_jsonl(preds, output_path)
