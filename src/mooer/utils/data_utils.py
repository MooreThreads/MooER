PROMPT_TEMPLATE_DICT = {
    'qwen': "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
}

PROMPT_DICT = {
    'asr': "Transcribe speech to text. ",
    'ast': "Translate speech to english text. ",
}