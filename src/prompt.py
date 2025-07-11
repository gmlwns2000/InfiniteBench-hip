import os

gpt4_templates = {
    "passkey": "There is an important info hidden inside a lot of irrelevant text. Find it and memorize them. I will quiz you about the important information there.\n\n{context}\n\n{input}",  # noqa
    "number_string": "There is an important info hidden inside a lot of irrelevant text. Find it. I will quiz you about the important information there.\n\n{context}\n\n{input}",  # noqa
    "kv_retrieval": "Extract the value corresponding to the specified key in the JSON object below.\n\n{context}\n\n{input}",  # noqa
    # "longbook_sum_eng": "Summarize the book below:\n\n{context}",  # noqa
    "longbook_qa_eng": "Read the book below and answer a question.\n\n{context}\n\nQuestion: {question}\n\nBe very concise.",  # noqa
    "longbook_choice_eng": "Read the book and answer the question.\n\n{context}\n\nQuestion: {question}\n\nOnly one of the following options is correct, tell me the answer using one single letter (A, B, C, or D). Don't say anything else.\nA. {OPTION_A}\nB. {OPTION_B}\nC. {OPTION_C}\nD. {OPTION_D}",  # noqa
    "longbook_sum_eng": "Summarize the following book.\n\n{context}",  # noqa
    "longbook_qa_chn": "请根据以下书籍回答我的问题。\n\n{context}\n\n问题：{question}\n请尽量简短地回答。",  # noqa
    "math_find": "{prefix}\n\n{context}\n\n{input}",
    "math_calc": "Compute the intermediate values in the following long expression.\n\n{context}",  # noqa
    "code_run": "Following is a set of Python functions. There is a function called named {func}.\n\n{context}\n\nPlease give me the exact number of the return value of {func_call}. Be concise. Your response must end with the final returned value.",  # noqa
    "code_debug": "There is ONLY ONE function in the large project that is deliberately made to include an obvious error. Please find the function that contains the most obvious errors. I will give you four options to narrow your scope. You can inspect the options and think. Eventually, tell me the answer using one single letter (A, B, C, or D).\n\n{context}\n\nWhich funtion has deliberate error?\nA. {OPTION_A}\nB. {OPTION_B}\nC. {OPTION_C}\nD. {OPTION_D}\n\nYou should first find the functions in the options. Repeat their content, inspect through code, and at last give me your answer for the function that has the deliberate and obvious error in A, B, C, or D.",  # noqa
    "longdialogue_qa_eng": "Below is a dialogue script where one random occurrence of a character name is replaced with \"$$MASK$$\", and you should try to guess who that character is.\n\nThe dialogue:\n\n---\n\n{context}\n\n---\n\nEnd of dialogue.\n\nWhich character is most likely \"$$MASK$$\"? Just say the name used by the scriptwriter (before the colon marks) of one single character and nothing else.",  # noqa
}

yarn_mistral_templates = {
    "passkey": "There is an important info hidden inside a lot of irrelevant text. Find it and memorize it. I will quiz you about the important information.\n\n{context}\n\n{input}\n\nThe pass key is",  # noqa
    "number_string": "There is an important info hidden inside a lot of irrelevant text. Find it. I will quiz you about the important information there.\n\n{context}\n\n{input}\n\nThe sequence of digits is",  # noqa
    "kv_retrieval": "Extract the value corresponding to the specified key in the JSON object below.\n\n{context}\n\n{input}",  # noqa
    "longbook_sum_eng": "Summarize the book below.\n\n{context}\n\nSummary:",  # noqa
    "longbook_choice_eng": "Read the book and answer the question.\n\n{context}\n\nQuestion: {question}\nA. {OPTION_A}\nB. {OPTION_B}\nC. {OPTION_C}\nD. {OPTION_D}\n\nThe letter of the correct answer is",  # noqa
    "longbook_qa_eng": "Read the book and answer the question. Be very concise in your answer.\n\n{context}\n\nQuestion: {question}\nAnswer:",  # noqa
    "longbook_qa_chn": "阅读以下书籍然后回答问题。\n\n{context}\n\n问题：{question}\n答案：",  # noqa
    "math_find": "{prefix}\n\n{context}\n\n{input}",
    "math_calc": "Let us calculate the intermediate values of an expression.\n\nExpression: 1 + 3 + 4\nValues: [1, 4, 8]\n\nExpression: 8 - 3 + 2 - 4\nValues: [8, 5, 7, 3]\n\nExpression: {context}\nValues:",  # noqa
    "code_run": "There is a function called {func} in the following Python code.\n\n{context}\n\nPlease compute the exact value of {func_call}. The value of {func_call} is",  # noqa
    "code_debug": "Following is a Python code where exactly one of the functions/methods has a deliberate error that makes it crash.\n\n{context}\n\nOptions:\nA. {OPTION_A}\nB. {OPTION_B}\nC. {OPTION_C}\nD. {OPTION_D}\n\nThe correct option is:",  # noqa
    "longdialogue_qa_eng": "Below is a dialogue script where one random occurrence of a character name is replaced with \"$$MASK$$\", and you should try to guess who that character is.\n\n{context}\n\nThe name that has been replaced with $$MASK$$ is likely",  # noqa
}

claude2_templates = {
    "passkey": "There is an important info hidden inside a lot of irrelevant text. Find it and memorize them. I will quiz you about the important information there.\n\n{context}\n{input}\nThe pass key is",
    "number_string": "There is an important info hidden inside a lot of irrelevant text. Find it. I will quiz you about the important information there.\n\n{context}\n{input}\nThe sequence of digits is",  # noqa
    "kv_retrieval": "There is an important info hidden inside a lot of irrelevant text. Find it. I will quiz you about the important information there.\n\n{context}\n{input}",
    "longbook_sum_eng": "Summarize the following book.\n\n{context}",  # noqa
    "longbook_choice_eng": "Read the book and answer the question.\n\n{context}\n\nQuestion: {question}\n\nOnly one of the following options is correct, tell me the answer using one single letter (A, B, C, or D). Don't say anything else.\nA. {OPTION_A}\nB. {OPTION_B}\nC. {OPTION_C}\nD. {OPTION_D}",  # noqa
    "longbook_qa_eng": "Read the novel below and answer a question:\n\n{context}\n\n{input}\nPlease answer as short as possible. The answer is: ",  # noqa
    "longbook_qa_chn": "请根据以下书籍回答我的问题。\n\n{context}\n\n问题：{question}\n请尽量简短地回答。",  # noqa
    "math_find": "{prefix}\n\n{context}\n\n{input}",
    "math_calc": "Let us calculate the intermediate values of an expression.\nExpression: 1 + 3 + 4\nValues: [1, 4, 8]\n\nExpression: 8 - 3 + 2 - 4\nValues: [8, 5, 7, 3]\n\nExpression: {context}\nValues:",  # noqa
    "code_run": "In the file functions_module.py, there is a function called ${func}.\n\n\nHere is the content of functions_module.py:\n{context}\n\nPlease give me the exact number of the return value of {func_call}. Your response should end with the sentence \'The return value is:\'.",  # noqa
    "code_debug": "There is ONLY ONE function in the large project that is deliberately made to include an obvious error. Please find the function that contains the most obvious errors. I will give you four options to narrow your scope. You can inspect through the options and think. Eventually, tell me the answer using one single letter (A, B, C, or D).\n\n{context}\n\nWhich funtion has deliberate error?\nA. {OPTION_A}\nB. {OPTION_B}\nC. {OPTION_C}\nD. {OPTION_D}\n\nYou should first find the functions in the options. Repeat their content, inspect through code, and at last give me your answer for the function that has the deliberate and obvious error in A, B, C, or D.",  # noqa
    "longdialogue_qa_eng": "Below is a dialogue script where one random occurrence of a character name is replaced with \"$$MASK$$\", and you should try to guess who that character is.\n\nThe dialogue:\n\n---\n\n{context}\n\n---\n\nEnd of dialogue.\n\nWhich character is most likely \"$$MASK$$\"? Just say the name used by the scriptwriter (before the colon marks) of one single character and nothing else.",  # noqa
}

kimi_templates = {
    "passkey": "There is an important info hidden inside a lot of irrelevant text. Find it and memorize them. I will quiz you about the important information there.\n\n{context}\n{input}\nThe pass key is",  # noqa
    "number_string": "There is an important info hidden inside a lot of irrelevant text. Find it. I will quiz you about the important information there.\n\n{context}\n{input}\nThe sequence of digits is",  # noqa
    "kv_retrieval": "Extract the value corresponding to the specified key in the JSON object below.\n\n{context}\n{input}",  # noqa
    "longbook_sum_eng": "Summarize the book below:\n\n{file:{context}}",  # noqa
    "longbook_choice_eng": "Read the book and answer the question.\n\nQuestion: {question}\n\nOnly one of the following options is correct, tell me the answer using one single letter (A, B, C, or D). Don't say anything else.\nA. {OPTION_A}\nB. {OPTION_B}\nC. {OPTION_C}\nD. {OPTION_D}" + "{file:{document}}",  # noqa
    "longbook_qa_eng": "Read the book below and answer a question.\n\nQuestion: {question}\n\nBe very concise." + "{file:{context}}",  # noqa
    "longbook_qa_chn": "阅读以下书籍然后回答问题。\n\n问题：{question}\n答案：" + "{file:{context}}",  # noqa
    "math_find": "{prefix}\n\n{context}\n\n{input}",
    "math_calc": "Let us calculate the intermediate values of an expression.\nExpression: 1 + 3 + 4\nValues: [1, 4, 8]\n\nExpression: 8 - 3 + 2 - 4\nValues: [8, 5, 7, 3]\n\nExpression: {context}\nValues:",  # noqa
    "code_run": "In the file functions_module.py, there is a function called ${func}.\n\n\nHere is the content of functions_module.py:\n\nPlease give me the exact number of the return value of ${func_call}. Your response should end with the sentence 'The return value is:'." + "{context}",  # noqa
    "code_debug": "Below is a code repository where there is one single function with bugs that causes an error. Please tell me the name of that function.\nWhich function has bugs? Give me the final answer in this format: \"[FINAL ANSWER: XXX]\". Don't say anything else." + "{fcontext}",  # noqa
    # "longdialogue_qa_eng": "Below is a dialogue script where one random occurrence of a character name is replaced with \"$$MASK$$\", and you should try to guess who that character is.\n\nThe name that has been replaced with $$MASK$$ is likely" + "{context}",  # noqa
    "longdialogue_qa_eng": "Below is a dialogue script where one random occurrence of a character name is replaced with \"$$MASK$$\", and you should try to guess who that character is. Give me the answer using the name before the colons, don't say anything else.\n\n{context}",  # noqa
}

IS_DEEPSEEK = os.getenv('IS_DEEPSEEK', '0') == '1'
IS_QWEN = os.getenv('IS_QWEN', '0') == '1'
IS_OPENROUTER = os.getenv('IS_OPENROUTER', '0') == '1'
IS_CHAT = os.getenv('IS_CHAT', '0') == '1'

if IS_OPENROUTER or IS_CHAT:
    LLAMA3_SYSTEM_PRM = """"""
    LLAMA3_USER_PRM = """"""
    LLAMA3_USER_END = """"""
elif IS_DEEPSEEK:
    # LLAMA3_SYSTEM_PRM = """<｜begin▁of▁sentence｜>"""
    LLAMA3_SYSTEM_PRM = """"""
    LLAMA3_USER_PRM = """<｜User｜>"""
    LLAMA3_USER_END = """<｜Assistant｜>"""
elif IS_QWEN:
    LLAMA3_SYSTEM_PRM = """<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"""
    LLAMA3_USER_PRM = """<|im_start|>user\n"""
    LLAMA3_USER_END = """<|im_end|>\n<|im_start|>assistant\n"""
else:
    LLAMA3_SYSTEM_PRM = """<|start_header_id|>system<|end_header_id|>

Cutting Knowledge Date: December 2023
Today Date: 26 Jul 2024

<|eot_id|>"""
    LLAMA3_USER_PRM = """<|start_header_id|>user<|end_header_id|>\n\n"""
    LLAMA3_USER_END = """\n\n<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"""

IS_INFLLM = os.getenv('IS_INFLLM', '0') == '1'

if not IS_INFLLM:
    llama3_templates = {
        # FIXED prompt
        "longbook_qa_eng": f"""{LLAMA3_SYSTEM_PRM}{LLAMA3_USER_PRM}You are a helpful assistant.

Your task is reading the book below (given text) and answer a question.

Here is my question: {{question}}

Be very concise.

--------------------------------------------------
- The start of given text.
--------------------------------------------------

{{context}}

--------------------------------------------------
- The end of given text.
--------------------------------------------------

Here, I repeat my question.

Question: {{question}}

Before answer the question, here is rules that you have to follow
- **Be very concise**.
- Answer within one sentence.
- You cannot answer like "there is no answer"
- All question have the answer.

Now, please answer to my question.{LLAMA3_USER_END}""",  # noqa
        "longbook_choice_eng": f"{LLAMA3_SYSTEM_PRM}{LLAMA3_USER_PRM}Read the book and answer the question.\n\nQuestion: {{question}}\n\nOnly one of the following options is correct, tell me the answer using one single letter (A, B, C, or D). Don't say anything else.\nA. {{OPTION_A}}\nB. {{OPTION_B}}\nC. {{OPTION_C}}\nD. {{OPTION_D}}\n\n-------{{context}}-------\n\nNow, answer to my question `{{question}}`, in single letter (A or B or C or D).\nI will repeat the options. \nA. {{OPTION_A}}\nB. {{OPTION_B}}\nC. {{OPTION_C}}\nD. {{OPTION_D}}\n\nNow, please answer to my question preciesly.\n\nQ: \"{{question}}\"\n\nThink carefully. {LLAMA3_USER_END}",  # noqa
        "longbook_sum_eng": f"{LLAMA3_SYSTEM_PRM}{LLAMA3_USER_PRM}Summarize the book below:\n\n-----{{context}}-----\n\nPlease summarize the given book.{LLAMA3_USER_END}",  # noqa
        "longdialogue_qa_eng": f"{LLAMA3_SYSTEM_PRM}{LLAMA3_USER_PRM}Below is a dialogue script where one random occurrence of a character name is replaced with \"$$MASK$$\", and you should try to guess who that character is. Give me the answer using the name before the colons, don't say anything else.\n\n-----{{context}}-----\n\nYou should try to guess $$MASK$$ who that character is. Give me the answer using the name before the colons, don't say anything else.{LLAMA3_USER_END}",  # noqa
        "longbook_qa_chn": f"{LLAMA3_SYSTEM_PRM}{LLAMA3_USER_PRM}阅读以下书籍然后回答问题。\n\n问题：{{question}}\n答案：{{context}}阅读以书籍然后回答问题。{LLAMA3_USER_END}",  # noqa
        
        "passkey": f"{LLAMA3_SYSTEM_PRM}{LLAMA3_USER_PRM}There is an important info hidden inside a lot of irrelevant text. Find it and memorize them. I will quiz you about the important information there.\n\n{{context}}\n{{input}}\n{LLAMA3_USER_END}The pass key is ",  # noqa
        "code_run": f"{LLAMA3_SYSTEM_PRM}{LLAMA3_USER_PRM}In the file functions_module.py, there is a function called ${{func}}.\n\n\nHere is the content of functions_module.py:\n\nPlease give me the exact number of the return value of ${{func_call}}. Your response should end with the sentence 'The return value is:'.\n\n{{context}}{LLAMA3_USER_END}",  # noqa
        "code_debug": f"{LLAMA3_SYSTEM_PRM}{LLAMA3_USER_PRM}Below is a code repository where there is one single function with bugs that causes an error. Please tell me the name of that function.\nWhich function has bugs? Give me the final answer in this format: \"[FINAL ANSWER: XXX]\". Don't say anything else.\n\n{{fcontext}}{LLAMA3_USER_END}",  # noqa
        "math_find": f"{LLAMA3_SYSTEM_PRM}{LLAMA3_USER_PRM}{{prefix}}\n\n{{context}}\n\n{{input}}{LLAMA3_USER_END}",
        "math_calc": f"{LLAMA3_SYSTEM_PRM}{LLAMA3_USER_PRM}Let us calculate the intermediate values of an expression.\nExpression: 1 + 3 + 4\nValues: [1, 4, 8]\n\nExpression: 8 - 3 + 2 - 4\nValues: [8, 5, 7, 3]\n\nExpression: {{context}}\n{LLAMA3_USER_END}Values:",  # noqa
        "number_string": f"{LLAMA3_SYSTEM_PRM}{LLAMA3_USER_PRM}There is an important info hidden inside a lot of irrelevant text. Find it. I will quiz you about the important information there.\n\n{{context}}\n{{input}}\n{LLAMA3_USER_END}The sequence of digits is",  # noqa
        "kv_retrieval": f"{LLAMA3_SYSTEM_PRM}{LLAMA3_USER_PRM}Extract the value corresponding to the specified key in the JSON object below.\n\n{{context}}\n{{input}}\n{LLAMA3_USER_END}",  # noqa
    }
else:
    llama3_templates = {
        "passkey": f"{LLAMA3_SYSTEM_PRM}{LLAMA3_USER_PRM}There is an important info hidden inside a lot of irrelevant text. Find it and memorize them. I will quiz you about the important information there.\n\n{{context}}\n\n{{input}}{LLAMA3_USER_END}The pass key is ",
        "all_passkey": f"{LLAMA3_SYSTEM_PRM}{LLAMA3_USER_PRM}There is an important info hidden inside a lot of irrelevant text. Find it and memorize them. I will quiz you about the important information there.\n\n{{context}}\n\n{{input}}{LLAMA3_USER_END}",
        "number_string": f"{LLAMA3_SYSTEM_PRM}{LLAMA3_USER_PRM}There is an important info hidden inside a lot of irrelevant text. Find it. I will quiz you about the important information there.\n\n{{context}}\n\n{{input}}{LLAMA3_USER_END}The sequence of digits mentioned in the text is ",
        "kv_retrieval": f"{LLAMA3_SYSTEM_PRM}{LLAMA3_USER_PRM}Extract the value corresponding to the specified key {{key}} in the JSON object below.\n\n{{context}}\n\n{{input}}\n\n(Hint: Answer only value. Do not answer anything else except the value of given key.){LLAMA3_USER_END}", 
        "longbook_qa_eng": f"{LLAMA3_SYSTEM_PRM}{LLAMA3_USER_PRM}Read the book below and answer a question.\n\nQuestion: {{question}}\n\nNow, I will provide the book content.\n\n=======\n\n{{context}}\n\n=======\n\nQuestion: {{question}}\n\nPlease answer as short as possible.{LLAMA3_USER_END}The answer is:", 
        "longbook_qa_eng_question_first": f"{LLAMA3_SYSTEM_PRM}{LLAMA3_USER_PRM}Read the book below and answer the question.\n\nQuestion: {{question}}\n\n{{context}}\n\nQuestion: {{question}}\n\nPlease answer as short as possible.{LLAMA3_USER_END}The answer is:", 
        "longbook_choice_eng": f"{LLAMA3_SYSTEM_PRM}{LLAMA3_USER_PRM}Read the book and answer the question.\n\n{{context}}\n\nQuestion: {{question}}\n\nOnly one of the following options is correct, tell me the answer using one single letter (A, B, C, or D). Don't say anything else.\nA. {{OPTION_A}}\nB. {{OPTION_B}}\nC. {{OPTION_C}}\nD. {{OPTION_D}}{LLAMA3_USER_END}",
        "longbook_sum_eng": f"{LLAMA3_SYSTEM_PRM}{LLAMA3_USER_PRM}Summarize the following book.\n\n{{context}}{LLAMA3_USER_END}",
        "longbook_qa_chn": f"{LLAMA3_SYSTEM_PRM}{LLAMA3_USER_PRM}请根据以下书籍回答我的问题。\n\n{{context}}\n\n问题：{{question}}\n请尽量简短地回答。{LLAMA3_USER_END}",
        "math_find": f"{LLAMA3_SYSTEM_PRM}{LLAMA3_USER_PRM}{{prefix}}\n\n{{context}}\n\n{{input}}{LLAMA3_USER_END}",
        "math_calc": f"{LLAMA3_SYSTEM_PRM}{LLAMA3_USER_PRM}Compute the intermediate values in the following long expression.\n\n{{context}}{LLAMA3_USER_END}",
        "code_run": f"{LLAMA3_SYSTEM_PRM}{LLAMA3_USER_PRM}Following is a set of Python functions. There is a function called named {{func}}.\n\n{{context}}\n\nPlease give me the exact number of the return value of {{func_call}}. Be concise. Your response must end with the final returned value.{LLAMA3_USER_END}",
        "code_debug": f"{LLAMA3_SYSTEM_PRM}{LLAMA3_USER_PRM}There is ONLY ONE function in the large project that is deliberately made to include an obvious error. Please find the function that contains the most obvious errors. I will give you four options to narrow your scope. You can inspect the options and think. Eventually, tell me the answer using one single letter (A, B, C, or D).\n\n{{context}}\n\nWhich funtion has deliberate error?\nA. {{OPTION_A}}\nB. {{OPTION_B}}\nC. {{OPTION_C}}\nD. {{OPTION_D}}\n\nGive me your answer for the function that has the deliberate and obvious error in A, B, C, or D. Your answer MUST be chosen from one of the four options without any explanation. If you cannot determine answers accurately, you also MUST provide the answer you think is most likely. Absolutely do not say you do not know or you need more information.{LLAMA3_USER_END}", 
        "longdialogue_qa_eng": f"{LLAMA3_SYSTEM_PRM}{LLAMA3_USER_PRM}Below is a dialogue script where one random occurrence of a character name is replaced with \"$$MASK$$\", and you should try to guess who that character is.\n\nThe dialogue:\n\n---\n\n{{context}}\n\n---\n\nEnd of dialogue.\n\nWhich character is most likely \"$$MASK$$\"? Just say the name used by the scriptwriter (before the colon marks) of one single character and nothing else.{LLAMA3_USER_END}"
    }
    if IS_CHAT:
        llama3_templates['passkey'] = f"{LLAMA3_SYSTEM_PRM}{LLAMA3_USER_PRM}There is an important info hidden inside a lot of irrelevant text. Find it and memorize them. I will quiz you about the important information there.\n\n{{context}}\n\n{{input}}\n\n(Instruction: You have to answer only the passkey. Do not answer anything else.)\n\n{LLAMA3_USER_END}"

EXAONE3_SYSTEM_PRM = """[BOS][|system|][|endofturn|]\n"""
EXAONE3_USER_PRM = """[|user|]\n"""
EXAONE3_USER_END = """[|endofturn|]\n[|assistant|]"""

exaone3_templates = {
    # FIXED prompt
    "longbook_qa_eng": f"""{EXAONE3_SYSTEM_PRM}{EXAONE3_USER_PRM}Read the book below and answer a question.

Question: {{question}}

Be very concise.

--------------------------------------------------
- The start of given text.
--------------------------------------------------

{{context}}

--------------------------------------------------
- The end of given text.
--------------------------------------------------

Here, I repeat my question. Question: {{question}}

Be very concise. Now, please answer to my question.{EXAONE3_USER_END}""",  # noqa
    "longbook_choice_eng": f"""{EXAONE3_SYSTEM_PRM}{EXAONE3_USER_PRM}Read the book and answer the question.
Question: {{question}}

Only one of the following options is correct, tell me the answer using one single letter (A, B, C, or D). Don't say anything else.
A. {{OPTION_A}}
B. {{OPTION_B}}
C. {{OPTION_C}}
D. {{OPTION_D}}

--------------------------------------------------
- The start of given text.
--------------------------------------------------

{{context}}

--------------------------------------------------
- The end of given text.
--------------------------------------------------

Now, answer to my question **{{question}}**, in single letter (A or B or C or D).

I will repeat the options.

A. {{OPTION_A}}
B. {{OPTION_B}}
C. {{OPTION_C}}
D. {{OPTION_D}}

Now, please answer to my question preciesly.

Q: \"{{question}}\"

Think carefully. Here is rules that you have to follow
- **Answer only single letter.** Do not say anything else.
- Do not explain the answer.
- For example, answer like "A." or "B." or "C." or "D."{EXAONE3_USER_END}""",  # noqa
    "longbook_sum_eng": f"{EXAONE3_SYSTEM_PRM}{EXAONE3_USER_PRM}Summarize the book below:\n\n-----{{context}}-----\n\nPlease summarize the given book.{EXAONE3_USER_END}",  # noqa
    "longdialogue_qa_eng": f"{EXAONE3_SYSTEM_PRM}{EXAONE3_USER_PRM}Below is a dialogue script where one random occurrence of a character name is replaced with \"$$MASK$$\", and you should try to guess who that character is. Give me the answer using the name before the colons, don't say anything else.\n\n-----{{context}}-----\n\nYou should try to guess $$MASK$$ who that character is. Give me the answer using the name before the colons, don't say anything else.{EXAONE3_USER_END}",  # noqa
    "longbook_qa_chn": f"{EXAONE3_SYSTEM_PRM}{EXAONE3_USER_PRM}阅读以下书籍然后回答问题。\n\n问题：{{question}}\n答案：{{context}}阅读以书籍然后回答问题。{EXAONE3_USER_END}",  # noqa
    
    "passkey": f"{EXAONE3_SYSTEM_PRM}{EXAONE3_USER_PRM}There is an important info hidden inside a lot of irrelevant text. Find it and memorize them. I will quiz you about the important information there.\n\n{{context}}\n{{input}}\n{EXAONE3_USER_END}The pass key is",  # noqa
    "code_run": f"{EXAONE3_SYSTEM_PRM}{EXAONE3_USER_PRM}In the file functions_module.py, there is a function called ${{func}}.\n\n\nHere is the content of functions_module.py:\n\nPlease give me the exact number of the return value of ${{func_call}}. Your response should end with the sentence 'The return value is:'.\n\n{{context}}{EXAONE3_USER_END}",  # noqa
    "code_debug": f"{EXAONE3_SYSTEM_PRM}{EXAONE3_USER_PRM}Below is a code repository where there is one single function with bugs that causes an error. Please tell me the name of that function.\nWhich function has bugs? Give me the final answer in this format: \"[FINAL ANSWER: XXX]\". Don't say anything else.\n\n{{fcontext}}{EXAONE3_USER_END}",  # noqa
    "math_find": f"{EXAONE3_SYSTEM_PRM}{EXAONE3_USER_PRM}{{prefix}}\n\n{{context}}\n\n{{input}}{EXAONE3_USER_END}",
    "math_calc": f"{EXAONE3_SYSTEM_PRM}{EXAONE3_USER_PRM}Let us calculate the intermediate values of an expression.\nExpression: 1 + 3 + 4\nValues: [1, 4, 8]\n\nExpression: 8 - 3 + 2 - 4\nValues: [8, 5, 7, 3]\n\nExpression: {{context}}\n{EXAONE3_USER_END}Values:",  # noqa
    "number_string": f"{EXAONE3_SYSTEM_PRM}{EXAONE3_USER_PRM}There is an important info hidden inside a lot of irrelevant text. Find it. I will quiz you about the important information there.\n\n{{context}}\n{{input}}\n{EXAONE3_USER_END}The sequence of digits is",  # noqa
    "kv_retrieval": f"{EXAONE3_SYSTEM_PRM}{EXAONE3_USER_PRM}Extract the value corresponding to the specified key in the JSON object below.\n\n{{context}}\n{{input}}\n{EXAONE3_USER_END}",  # noqa
}

GEMMA2_SYSTEM_PRM = """<bos>"""
GEMMA2_USER_PRM = """<start_of_turn>user\n"""
GEMMA2_USER_END = """<end_of_turn>\n<start_of_turn>assistant\n"""

gemma2_templates = {
    # FIXED prompt
    "longbook_qa_eng": f"{GEMMA2_SYSTEM_PRM}{GEMMA2_USER_PRM}Read the book below and answer a question.\n\nQuestion: {{question}}\n\nBe very concise.-----{{context}}-----\n\nHere, I repeat my question. Question: {{question}}\n\nBe very concise. Now, please answer to my question.{GEMMA2_USER_END}",  # noqa
    "longbook_choice_eng": f"{GEMMA2_SYSTEM_PRM}{GEMMA2_USER_PRM}Read the book and answer the question.\n\nQuestion: {{question}}\n\nOnly one of the following options is correct, tell me the answer using one single letter (A, B, C, or D). Don't say anything else.\nA. {{OPTION_A}}\nB. {{OPTION_B}}\nC. {{OPTION_C}}\nD. {{OPTION_D}}\n\n-------{{context}}-------\n\nNow, answer to my question `{{question}}`, in single letter (A or B or C or D).\n\nI will repeat the options.\n\nA. {{OPTION_A}}\nB. {{OPTION_B}}\nC. {{OPTION_C}}\nD. {{OPTION_D}}\n\nNow, please answer to my question preciesly.\n\nQ: \"{{question}}\"\n\nThink carefully. **Answer only single letter.** You do not need to explain anything else.{GEMMA2_USER_END}",  # noqa
    "longbook_sum_eng": f"{GEMMA2_SYSTEM_PRM}{GEMMA2_USER_PRM}Summarize the book below:\n\n-----{{context}}-----\n\nPlease summarize the given book.{GEMMA2_USER_END}",  # noqa
    "longdialogue_qa_eng": f"{GEMMA2_SYSTEM_PRM}{GEMMA2_USER_PRM}Below is a dialogue script where one random occurrence of a character name is replaced with \"$$MASK$$\", and you should try to guess who that character is. Give me the answer using the name before the colons, don't say anything else.\n\n-----{{context}}-----\n\nYou should try to guess $$MASK$$ who that character is. Give me the answer using the name before the colons, don't say anything else.{GEMMA2_USER_END}",  # noqa
    "longbook_qa_chn": f"{GEMMA2_SYSTEM_PRM}{GEMMA2_USER_PRM}阅读以下书籍然后回答问题。\n\n问题：{{question}}\n答案：{{context}}阅读以书籍然后回答问题。{GEMMA2_USER_END}",  # noqa
    
    "passkey": f"{GEMMA2_SYSTEM_PRM}{GEMMA2_USER_PRM}There is an important info hidden inside a lot of irrelevant text. Find it and memorize them. I will quiz you about the important information there.\n\n{{context}}\n{{input}}\n{GEMMA2_USER_END}The pass key is",  # noqa
    "code_run": f"{GEMMA2_SYSTEM_PRM}{GEMMA2_USER_PRM}In the file functions_module.py, there is a function called ${{func}}.\n\n\nHere is the content of functions_module.py:\n\nPlease give me the exact number of the return value of ${{func_call}}. Your response should end with the sentence 'The return value is:'.\n\n{{context}}{GEMMA2_USER_END}",  # noqa
    "code_debug": f"{GEMMA2_SYSTEM_PRM}{GEMMA2_USER_PRM}Below is a code repository where there is one single function with bugs that causes an error. Please tell me the name of that function.\nWhich function has bugs? Give me the final answer in this format: \"[FINAL ANSWER: XXX]\". Don't say anything else.\n\n{{fcontext}}{GEMMA2_USER_END}",  # noqa
    "math_find": f"{GEMMA2_SYSTEM_PRM}{GEMMA2_USER_PRM}{{prefix}}\n\n{{context}}\n\n{{input}}{GEMMA2_USER_END}",
    "math_calc": f"{GEMMA2_SYSTEM_PRM}{GEMMA2_USER_PRM}Let us calculate the intermediate values of an expression.\nExpression: 1 + 3 + 4\nValues: [1, 4, 8]\n\nExpression: 8 - 3 + 2 - 4\nValues: [8, 5, 7, 3]\n\nExpression: {{context}}\n{GEMMA2_USER_END}Values:",  # noqa
    "number_string": f"{GEMMA2_SYSTEM_PRM}{GEMMA2_USER_PRM}There is an important info hidden inside a lot of irrelevant text. Find it. I will quiz you about the important information there.\n\n{{context}}\n{{input}}\n{GEMMA2_USER_END}The sequence of digits is",  # noqa
    "kv_retrieval": f"{GEMMA2_SYSTEM_PRM}{GEMMA2_USER_PRM}Extract the value corresponding to the specified key in the JSON object below.\n\n{{context}}\n{{input}}\n{GEMMA2_USER_END}",  # noqa
}

MISTRAL_SYSTEM_PRM = """<s>"""
MISTRAL_USER_PRM = """ [INST] """
MISTRAL_USER_END = """ [/INST]"""

mistral_templates = {
    "passkey": f"{MISTRAL_SYSTEM_PRM}{MISTRAL_USER_PRM}There is an important info hidden inside a lot of irrelevant text. Find it and memorize them. I will quiz you about the important information there.\n\n{{context}}\n\n{{input}}{MISTRAL_USER_END}The pass key is ",
    "all_passkey": f"{MISTRAL_SYSTEM_PRM}{MISTRAL_USER_PRM}There is an important info hidden inside a lot of irrelevant text. Find it and memorize them. I will quiz you about the important information there.\n\n{{context}}\n\n{{input}}{MISTRAL_USER_END}",
    "number_string": f"{MISTRAL_SYSTEM_PRM}{MISTRAL_USER_PRM}There is an important info hidden inside a lot of irrelevant text. Find it. I will quiz you about the important information there.\n\n{{context}}\n\n{{input}}{MISTRAL_USER_END}The sequence of digits mentioned in the text is ",
    "kv_retrieval": f"{MISTRAL_SYSTEM_PRM}{MISTRAL_USER_PRM}Extract the value corresponding to the specified key {{key}} in the JSON object below.\n\n{{context}}\n\n{{input}}{MISTRAL_USER_END}", 
    "longbook_qa_eng": f"{MISTRAL_SYSTEM_PRM}{MISTRAL_USER_PRM}Read the book below and answer a question.\n\nQuestion: {{question}}\n\nNow, I will provide the book content.\n\n=======\n\n{{context}}\n\n=======\n\nQuestion: {{question}}\n\nPlease answer as short as possible. The answer is:{MISTRAL_USER_END}", 
    "longbook_qa_eng_question_first": f"{MISTRAL_SYSTEM_PRM}{MISTRAL_USER_PRM}Read the book below and answer the question.\n\nQuestion: {{question}}\n\n{{context}}\n\nQuestion: {{question}}\n\nPlease answer as short as possible. The answer is:{MISTRAL_USER_END}", 
    "longbook_choice_eng": f"{MISTRAL_SYSTEM_PRM}{MISTRAL_USER_PRM}Read the book and answer the question.\n\n{{context}}\n\nQuestion: {{question}}\n\nOnly one of the following options is correct, tell me the answer using one single letter (A, B, C, or D). Don't say anything else.\nA. {{OPTION_A}}\nB. {{OPTION_B}}\nC. {{OPTION_C}}\nD. {{OPTION_D}}{MISTRAL_USER_END}",
    "longbook_sum_eng": f"{MISTRAL_SYSTEM_PRM}{MISTRAL_USER_PRM}Summarize the following book.\n\n{{context}}{MISTRAL_USER_END}",
    "longbook_qa_chn": f"{MISTRAL_SYSTEM_PRM}{MISTRAL_USER_PRM}请根据以下书籍回答我的问题。\n\n{{context}}\n\n问题：{{question}}\n请尽量简短地回答。{MISTRAL_USER_END}",
    "math_find": f"{MISTRAL_SYSTEM_PRM}{MISTRAL_USER_PRM}{{prefix}}\n\n{{context}}\n\n{{input}}{MISTRAL_USER_END}",
    "math_calc": f"{MISTRAL_SYSTEM_PRM}{MISTRAL_USER_PRM}Compute the intermediate values in the following long expression.\n\n{{context}}{MISTRAL_USER_END}",
    "code_run": f"{MISTRAL_SYSTEM_PRM}{MISTRAL_USER_PRM}Following is a set of Python functions. There is a function called named {{func}}.\n\n{{context}}\n\nPlease give me the exact number of the return value of {{func_call}}. Be concise. Your response must end with the final returned value.{MISTRAL_USER_END}",
    "code_debug": f"{MISTRAL_SYSTEM_PRM}{MISTRAL_USER_PRM}There is ONLY ONE function in the large project that is deliberately made to include an obvious error. Please find the function that contains the most obvious errors. I will give you four options to narrow your scope. You can inspect the options and think. Eventually, tell me the answer using one single letter (A, B, C, or D).\n\n{{context}}\n\nWhich funtion has deliberate error?\nA. {{OPTION_A}}\nB. {{OPTION_B}}\nC. {{OPTION_C}}\nD. {{OPTION_D}}\n\nGive me your answer for the function that has the deliberate and obvious error in A, B, C, or D. Your answer MUST be chosen from one of the four options without any explanation. If you cannot determine answers accurately, you also MUST provide the answer you think is most likely. Absolutely do not say you do not know or you need more information.{MISTRAL_USER_END}", 
    "longdialogue_qa_eng": f"{MISTRAL_SYSTEM_PRM}{MISTRAL_USER_PRM}Below is a dialogue script where one random occurrence of a character name is replaced with \"$$MASK$$\", and you should try to guess who that character is.\n\nThe dialogue:\n\n---\n\n{{context}}\n\n---\n\nEnd of dialogue.\n\nWhich character is most likely \"$$MASK$$\"? Just say the name used by the scriptwriter (before the colon marks) of one single character and nothing else.{MISTRAL_USER_END}"
}