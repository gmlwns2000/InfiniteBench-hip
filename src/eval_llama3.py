import json
import os
from pathlib import Path
import queue
import time
import traceback
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

IS_QUESTION = os.getenv('IS_QUESTION', '0') == '1'
IS_THINK = os.getenv('IS_THINK', '0') == '1'
IS_CHAT = os.getenv('IS_CHAT', '0') == '1'
IS_OPENROUTER = os.getenv('IS_OPENROUTER', '0') == '1'
OPENROUTER_KEY = os.getenv('OPENROUTER_KEY', 'none')
OPENROUTER_MODEL = os.getenv('OPENROUTER_MODEL', 'google/gemini-flash-1.5')
IS_DEEPSEEK = os.getenv('IS_DEEPSEEK', '0') == '1'
USING_SGLANG = os.getenv('USING_SGLANG', '0') == '1'
SGLANG_PORT = int(os.getenv('SGLANG_PORT', '30000'))
SGLANG_MODEL = os.getenv('SGLANG_MODEL', 'anything')
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
        
        if USING_SGLANG:
            import requests

            prompt_text = tok.decode(input_ids[0], skip_special_tokens=False)
            
            if IS_DEEPSEEK:
                task_max_tokens = max_tokens
                sampling_params = {
                    "top_p": 1.0,
                    "temperature": 0.4,
                    "max_new_tokens": 8192,
                }
            else:
                if IS_THINK:
                    max_tokens = 8192
                    task_max_tokens = max_tokens
                    
                    sampling_params = {
                        "top_k": 20 ,
                        "temperature": 0.6,
                        "top_p": 0.95,
                        "max_new_tokens": max_tokens,
                        "max_tokens": max_tokens,
                    }
                else:
                    sampling_params = {
                        # "top_k": 1, # greedy
                        "temperature": 0.0,
                        "top_p": 1e-6,
                        "max_new_tokens": max_tokens,
                        "max_tokens": max_tokens,
                    }
                    sampling_params.update({
                        "chat_template_kwargs": {"enable_thinking": False}
                    })
            
            if IS_CHAT:
                if not IS_QUESTION:
                    response = requests.post(
                        f"http://localhost:{SGLANG_PORT}/v1/chat/completions",
                        json={
                            "model": SGLANG_MODEL,
                            "messages": [{
                                'role': 'user', 
                                'content': prompt_text
                            }],
                            **sampling_params,
                        },
                    )
                    assert response.status_code == 200, response.json()
                    
                    decoded = response.json()["choices"][0]["message"]["content"] # type: str
                else:
                    response = requests.post(
                        f"http://localhost:{SGLANG_PORT}/v1/chat/completions",
                        json={
                            "model": SGLANG_MODEL,
                            "messages": [
                                {
                                    'role': 'system',
                                    'content': r"Your job is scanning user's message and answer the potential questions of user including current message."
                                },
                                {
                                    'role': 'user', 
                                    'content': (
                                        f"Here is user's message. Please read carefully and think potential user's questions."
                                        f"\n\n-----\n\n{prompt_text}\n\n-----\n\n"
                                    )
                                },
                            ],
                            "top_p": 1e-6,
                            "max_new_tokens": 1024,
                        },
                    )
                    assert response.status_code == 200, response.json()
                    
                    decoded = response.json()["choices"][0]["message"]["content"] # type: str
                    print(decoded)
                    
                    response = requests.post(
                        f"http://localhost:{SGLANG_PORT}/v1/chat/completions",
                        json={
                            "model": SGLANG_MODEL,
                            "messages": [
                                {
                                    'role': 'system',
                                    'content': (
                                        f"You are a helpful assistant. Your job is answering user's query.\n\n"
                                        f"Your kind superviosor already collect user's potential queries.\n\n"
                                        f"Provided questions may related to upcoming user's behavior, so you need to remember them carefully.\n\n"
                                        f"Here is collected potential questions from users:\n\n"
                                        f"```py\n{decoded}```\n\n"
                                        f"Now, read carefully to user's message and please answer user's query."
                                    )
                                },
                                {
                                    'role': 'user', 
                                    'content': prompt_text
                                },
                            ],
                            **sampling_params,
                        },
                    )
                    assert response.status_code == 200, response.json()
                    
                    decoded = response.json()["choices"][0]["message"]["content"] # type: str
                    
                    print(decoded)
            else:
                if 'max_tokens' in sampling_params:
                    del sampling_params['max_tokens']
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
            
            if IS_DEEPSEEK or IS_THINK:
                # print(f'RAW decoded: \n--------------\n{decoded}\n--------------\n')
                # print('</think>' in decoded, decoded.find('</think>'))
                if '</think>' in decoded:
                    start_idx = decoded.find('</think>')
                    decoded = decoded[start_idx + len('</think>'):].strip()
                else:
                    decoded = decoded[-task_max_tokens*3:]
            
            responses = [decoded]
        elif IS_OPENROUTER:
            from openai import OpenAI
            client = OpenAI(
                base_url='https://openrouter.ai/api/v1',
                api_key=OPENROUTER_KEY,
            )
            
            while True:
                prompt_text = tok.decode(input_ids[0], skip_special_tokens=False)
                
                try:
                    completion = client.chat.completions.create(
                        model=OPENROUTER_MODEL,
                        messages=[
                            {
                                'role': 'user',
                                'content': [
                                    {
                                        'type': 'text',
                                        'text': prompt_text,
                                    }
                                ]
                            }
                        ]
                    )
                except Exception as ex:
                    print(ex)
                    time.sleep(1)
                    continue
                
                print(completion)
                
                if completion.choices is not None:                
                    responses = [
                        completion.choices[0].message.content
                    ]
                    break
                else:
                    print('retry')
                    time.sleep(1)
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
    print_output: bool = True,
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
    if print_output:
        print("Chunked generation:", output.replace('\n', '\\n'))
    return output, len_after


from hip_attn.models.modeling_llama import LlamaForCausalLM
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
    if not (USING_SGLANG or IS_OPENROUTER):
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
    
    pool_size = int(os.getenv('JOBS', '1'))
    
    if pool_size == 1:
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
    else:    
        lock = threading.Lock()
        job_queue = queue.Queue(maxsize=pool_size * 2)
        result_queue = queue.Queue()
        def pool_main():
            while True:
                job = job_queue.get()
                if job is None:
                    return

                try:
                    eg = examples[job]
                    while len(input_texts) <= job:
                        time.sleep(0.001)
                    input_text = input_texts[job]
                    print(f'[{threading.current_thread().native_id}] Send example {job}.')
                    pred, len_after = get_pred(
                        model, tok, input_text,
                        max_tokens=max_tokens,
                        verbose=args.verbose,
                        reqiure_truncate=False,
                        print_output=False,
                    )
                    pred_to_print = pred.replace('\n', '\\n')
                    print(f'[{threading.current_thread().native_id}] Generated example {job}: {pred_to_print}')
                    if args.verbose:
                        print(pred)
                    result = {
                        "id": i,
                        "prediction": pred,
                        "ground_truth": get_answer(eg, data_name),
                        "context_length": len_after,
                    }
                    result_queue.put((job, result))
                except Exception as ex:
                    traceback.print_exc()
                
        thread_pool = [threading.Thread(target=pool_main, daemon=True) for i in range(pool_size)]
        for t in thread_pool:
            t.start()
        
        for i in range(args.start_idx, args.stop_idx):
            # eg = examples[i]
            # print(f"====== Example {i} ======")
            # while len(input_texts) <= i:
            #     time.sleep(0.001)
            # input_text = input_texts[i]
            # pred, len_after = get_pred(
            #     model, tok, input_text,
            #     max_tokens=max_tokens,
            #     verbose=args.verbose,
            #     reqiure_truncate=False,
            # )
            # if args.verbose:
            #     print(pred)
            # preds.append(
            #     {
            #         "id": i,
            #         "prediction": pred,
            #         "ground_truth": get_answer(eg, data_name),
            #         "context_length": len_after,
            #     }
            # )
            # dump_jsonl(preds, output_path)
            
            if not job_queue.full():
                print('enqueu', i)
                job_queue.put(i)
            else:
                print('fdfasdf', flush=True)
                max_get = pool_size
                while not result_queue.empty():
                    print('asdf', flush=True)
                    max_get -= 1
                    if max_get < 0: break
                    print('asdf4', flush=True)
                    job_id, result = result_queue.get(i)
                    print('asdf3', flush=True)
                    preds.append(result)
                    print('asdf1', flush=True)
                    dump_jsonl(preds, output_path)

        print('asdf', flush=True)
        while not result_queue.empty():
            print('asdf4', flush=True)
            job_id, result = result_queue.get(i)
            print('asdf3', flush=True)
            preds.append(result)
            print('asdf1', flush=True)
            dump_jsonl(preds, output_path)
        
        for i in range(pool_size):
            job_queue.put(None)
        for t in thread_pool:
            t.join()