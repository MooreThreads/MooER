import time
import sox
import torch
try:
    import torch_musa
except ImportError as e:
    print("You should install torch_musa if you want to run on Moore Threads GPU")
import os
import argparse
import torchaudio
from torchaudio.transforms import Resample
import logging
from mooer.datasets.speech_processor import *
from mooer.configs import asr_config
from mooer.models import mooer_model
from mooer.utils.utils import *
import gradio as gr
from transformers import TextIteratorStreamer
from threading import Thread

parser = argparse.ArgumentParser()
parser.add_argument("--task", default='mtl', choices=['asr', 'ast', 'mtl'], type=str, help="task: asr or ast. Please set ast if you choose a asr/ast multitask model")
parser.add_argument("--cmvn_path", default='', type=str, help="cmvn path. If not set, will use path in src/mooer/configs/asr_config.py")
parser.add_argument("--encoder_path", default='', type=str, help="encoder path. If not set, will use the path in src/mooer/configs/asr_config.py")
parser.add_argument("--llm_path", default='', type=str, help="llm path. If not set, will use the path in src/mooer/configs/asr_config.py")
parser.add_argument("--adapter_path", default='pretrained_models/asr_ast_mtl/adapter_project.pt', type=str, help="asr/ast multitask adapter path.")
parser.add_argument("--lora_dir", default='pretrained_models/asr_ast_mtl/lora_weights', type=str, help="asr/ast multitask lora path.")
parser.add_argument("--server_port", default=10010, type=int, help="gradio server port")
parser.add_argument("--server_name", default="0.0.0.0", type=str, help="gradio server name")
parser.add_argument("--share", default=False, type=lambda x: (str(x).lower() == 'true'), help="whether to share the server to public")
args = parser.parse_args()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    filemode='w'
)

logger = logging.getLogger()
logger.setLevel(logging.INFO)

PROMPT_TEMPLATE_DICT = {
    'qwen': "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
}
PROMPT_DICT = {
    'asr': "Transcribe speech to text. ",
    'ast': "Translate speech to english text. ",
}

global_task = args.task
model_config = {
    global_task: asr_config.ModelConfig(),
}

if args.llm_path and os.path.exists(args.llm_path):
    model_config[global_task].llm_path = args.llm_path

if args.cmvn_path and os.path.exists(args.cmvn_path):
    model_config[global_task].cmvn_path = args.cmvn_path

if args.encoder_path and os.path.exists(args.encoder_path):
    model_config[global_task].encoder_path = args.encoder_path

if args.adapter_path and os.path.exists(args.adapter_path):
    model_config[global_task].adapter_path = args.adapter_path
if args.lora_dir and os.path.exists(args.lora_dir):
    model_config[global_task].lora_dir = args.lora_dir

if args.task:
    model_config[global_task].prompt_key = 'ast' if args.task == 'mtl' else args.task

device = str(get_device())
logger.info("This demo will run on {}".format(device.upper()))


model = {}
for index, task in enumerate(model_config):
    logger.info(model_config[task])
    this_model, this_tokenizer = mooer_model.init_model(
        model_config=model_config[task])
    model[task] = {
        "model": this_model,
        "tokenizer": this_tokenizer
    }
    model[task]['model'].to(device+f':{index}')
    model[task]['model'].eval()
    model[task]['device'] = device+f':{index}'

# shared models and parameters
prompt_template_key = model_config[global_task].get('prompt_template_key', 'qwen')
prompt_template = PROMPT_TEMPLATE_DICT[prompt_template_key]
prompt_key = model_config[global_task].get('prompt_key', 'asr')
prompt_org = PROMPT_DICT[prompt_key]
cmvn = load_cmvn(model_config[global_task].get('cmvn_path'))
adapter_downsample_rate = model_config[global_task].get('adapter_downsample_rate')
logger.info(f"Use LLM Type {prompt_template_key}, "
            f"Prompt template {prompt_template}, "
            f"Use task type {prompt_key}, "
            f"Prompt {prompt_org}")

load_dtype = model_config[global_task].get('load_dtype', 'bfloat16')
dtype = torch.float32
if load_dtype == 'float16':
    dtype = torch.float16
elif load_dtype == 'bfloat16':
    dtype = torch.bfloat16
logging.info(f"Input data type: {dtype}")

context_scope = torch.musa.amp.autocast if 'musa' in device else torch.cuda.amp.autocast

def convert(inputfile, outfile):
    sox_tfm = sox.Transformer()
    sox_tfm.set_output_format(
            file_type="wav", channels=1, encoding="signed-integer", rate=16000, bits=16
    )
    sox_tfm.build(inputfile, outfile)

def process_wav(task, wav_path):
    audio_raw, sample_rate = torchaudio.load(wav_path)
    assert sample_rate == 16000 and audio_raw.shape[0] == 1

    audio_raw = audio_raw[0]
    duration = audio_raw.shape[0] / 16000.
    prompt = prompt_template.format(prompt_org)
    audio_mel = compute_fbank(waveform=audio_raw)
    audio_mel = apply_lfr(inputs=audio_mel, lfr_m=7, lfr_n=6)
    audio_mel = apply_cmvn(audio_mel, cmvn=cmvn)
    audio_length = audio_mel.shape[0]
    audio_length = audio_length // adapter_downsample_rate
    audio_pseudo = torch.full((audio_length,), -1)
    prompt_ids = model[task]["tokenizer"].encode(prompt)
    prompt_length = len(prompt_ids)
    prompt_ids = torch.tensor(prompt_ids, dtype=torch.int64)
    example_ids = torch.cat((audio_pseudo, prompt_ids))  # [audio, prompt]
    example_mask = example_ids.ge(-1)
    
    items = {
        "input_ids": example_ids,
        "attention_mask": example_mask,
        "audio_mel": audio_mel,
        "audio_length": audio_length,
        "prompt_length": prompt_length,
        "duration": duration,
    }
    return items


def unify_forward(task, audio_file):
    this_tokenizer = model[task]['tokenizer']
    this_model = model[task]['model']
    this_device = model[task]['device']
    overall_st = time.time()
    with torch.no_grad():
        st = time.time()
        convert(audio_file, audio_file + '.16k.wav')
        audio_file = audio_file + '.16k.wav'
        items = process_wav(task, audio_file)
        et = time.time()
        logger.info(f"Process wav takes {et - st}s")
        st = time.time()
        batch = process_batch([items], tokenizer=this_tokenizer)
        et = time.time()
        logger.info(f"Process batch takes {et - st}s")
        st = time.time()
        for key in batch.keys():
            batch[key] = batch[key].to(this_device) if isinstance(batch[key], torch.Tensor) else batch[key]
        with context_scope(dtype=dtype):
            model_outputs = this_model.generate(**batch)
        et = time.time()
        logger.info(f"Forward takes {et - st}s")
        st = time.time()
        output_text = this_model.tokenizer.batch_decode(
            model_outputs, add_special_tokens=False, skip_special_tokens=True)
        et = time.time()
        logger.info(f"Decode takes {et - st}s")
        asr_text = ''
        ast_text = ''
        for text in output_text:
            if task == 'asr':
                asr_text = text
                ast_text = ''
            elif task == 'ast':
                asr_text = ''
                ast_text = text
            elif task == 'mtl':
                if '\n' in text:
                    asr_text = text.split('\n')[0]
                    ast_text = text.split('\n')[1]
                else:
                    asr_text = text
                    ast_text = ''
    overall_et = time.time()
    logger.info("Cost {}s to do the inference.".format(overall_et - overall_st))
    return asr_text, ast_text


def mtl_inference(mic_input, file_input):
    task = global_task
    try:
        if mic_input is not None:
            asr_res, ast_res = unify_forward(task, mic_input)
        elif file_input is not None:
            asr_res, ast_res = unify_forward(task, file_input)
        else:
            logger.info("Empty input")
            return '', ''
        return asr_res, ast_res
    except Exception as e:
        logger.error(e)
        return '', ''
    

def unify_forward_stream(task, audio_file):
    this_tokenizer = model[task]['tokenizer']
    this_model = model[task]['model']
    this_device = model[task]['device']
    with torch.no_grad():
        convert(audio_file, audio_file + '.16k.wav')
        audio_file = audio_file + '.16k.wav'
        items = process_wav(task, audio_file)
        batch = process_batch([items], tokenizer=this_tokenizer)
        for key in batch.keys():
            batch[key] = batch[key].to(this_device) if isinstance(batch[key], torch.Tensor) else batch[key]
        with context_scope(dtype=dtype):
            inputs_embeds, attention_mask, kwargs = this_model.generate(**batch, compute_llm=False)
            streamer = TextIteratorStreamer(
                tokenizer=this_tokenizer, timeout=60.0, skip_prompt=True, skip_special_tokens=True
            )
            
            def generate_and_signal_complete():
                this_model.llm.generate(
                    inputs_embeds=inputs_embeds,
                    max_new_tokens=kwargs.get("max_new_tokens", 500),
                    num_beams=kwargs.get("num_beams", 1),
                    do_sample=kwargs.get("do_sample", False),
                    min_length=kwargs.get("min_length", 1),
                    top_p=kwargs.get("top_p", 1.0),
                    repetition_penalty=kwargs.get("repetition_penalty", 1.0),
                    length_penalty=kwargs.get("length_penalty", 1.0),
                    temperature=kwargs.get("temperature", 1.0),
                    attention_mask=attention_mask,
                    bos_token_id=this_model.tokenizer.bos_token_id,
                    eos_token_id=this_model.tokenizer.eos_token_id,
                    pad_token_id=this_model.tokenizer.pad_token_id,
                    streamer=streamer
                )
            
            t1 = Thread(target=generate_and_signal_complete)
            t1.start()
        
        partial_text = ""
        for new_text in streamer:
            partial_text += new_text
            if task == 'asr':
                asr_text = partial_text
                ast_text = ''
            elif task == 'ast':
                asr_text = ''
                ast_text = partial_text
            elif task == 'mtl':
                if '\n' in partial_text:
                    asr_text = partial_text.split('\n')[0]
                    ast_text = partial_text.split('\n')[1]
                else:
                    asr_text = partial_text
                    ast_text = ''
            yield asr_text, ast_text


def mtl_inference_stream(mic_input, file_input):
    task = global_task
    try:
        if mic_input is not None:
            yield from unify_forward_stream(task, mic_input)
        elif file_input is not None:
            yield from unify_forward_stream(task, file_input)
        else:
            logger.info("Empty input")
            return '', ''
    except Exception as e:
        logger.error(e)
        return '', ''


logo = '''
    <div style="width: 130px;">
      <img src="https://mt-ai-speech-public.tos-cn-beijing.volces.com/MTLogo.png" width="130">
    </div>
'''

description = '''
    # MooER 摩耳

    *MooER* is an LLM-based speech recognition/translation model capable of transcribing input speech into text and translating it into another language in an end-to-end manner.
    For more details, please refer to [the repo](https://github.com/MooreThreads/MooER).

    Please note that the current version DOES NOT SUPPORT mobile phones. Use your PC or Mac instead.
'''

with gr.Blocks(title="MooER online demo") as interface:
    gr.HTML(logo)
    gr.Markdown(description)
    with gr.Row():
        mic_input = gr.Audio(
                    sources='microphone',
                    type="filepath",
                    label="record your voice",
                    show_download_button=True,
                )
        file_input = gr.Audio(sources="upload", type="filepath", label="upload a file", show_download_button=True)
    with gr.Column():
        text_output_asr = gr.Textbox(label="Speech Recognition", lines=3, max_lines=10)
        text_output_ast = gr.Textbox(label="Speech Translation", lines=3, max_lines=10)
    
    with gr.Row():
        mtl_btn = gr.Button("Transcribe / Translate")
        mtl_btn_stream = gr.Button("Transcribe / Translate in streaming mode. Faster but less accurate.")
    mtl_btn.click(fn=mtl_inference, inputs=[mic_input, file_input], outputs=[text_output_asr, text_output_ast], concurrency_id="mtl")
    mtl_btn_stream.click(fn=mtl_inference_stream, inputs=[mic_input, file_input], outputs=[text_output_asr, text_output_ast], concurrency_id="mtl")

interface.queue().launch(
    favicon_path='demo/resources/mt_favicon.png',
    server_name=args.server_name,
    server_port=args.server_port,
    share=args.share
)

