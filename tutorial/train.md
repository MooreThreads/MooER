# MooER 训练教程

## 目录
1. [简介](#简介)
2. [环境准备](#环境准备)
3. [数据打包](#数据打包)
   - [数据准备](#数据准备)
   - [打包流程](#打包流程)
4. [模型训练](#模型训练)
   - [配置文件](#配置文件)
   - [训练流程](#训练流程)
5. [解码与推理](#解码与推理)
   - [解码方法](#解码方法)
   - [推理示例](#推理示例)
   - [结果分析](#结果分析)  

---

## 简介
摩耳大模型（英文名：MooER）—— 一个由摩尔线程开发的、基于大语言模型（Large Language Model，LLM）的语音识别和语音翻译系统。通过摩耳框架，您可以基于大语言模型，以端到端的方式，将输入语音自动转录为文本（即语音识别），并将其翻译为其它语言（即语音翻译）。也可以轻易基于本框架完成其他您想完成的音频理解任务的训练和推理。本教程基于MT5K（5000h训练数据规模）的ASR模型的训练参数和配置，囊括了***训练数据处理 -> 模型训练 -> 模型测试*** 。您也可以基于我们开源的8万小时训练的基础模型进行微调。我们基于MT5K，在开源的训练和解码框架下，多个测试集效果对齐了我们的内部代码训练效果。

## 环境准备
参考 https://github.com/MooreThreads/MooER/blob/master/README.md#%EF%B8%8F-build-environtment 部分

## 数据打包

### 数据准备
需要准备你的音频、和对应训练的标签，例如ASR的音频和对应的文本。音频文件应采用 `.wav` 格式，采样率为 16kHz，单通道。`wav.scp` 文件用于记录音频文件的UTTID及其路径，以便在训练时能够正确定位音频文件。

**`wav.scp` 样例：**

```plaintext
BAC009S0764W0121 /nfs2/speech/data/asr/aishell/data_aishell/wav/test/S0764/BAC009S0764W0121.wav
BAC009S0764W0122 /nfs2/speech/data/asr/aishell/data_aishell/wav/test/S0764/BAC009S0764W0122.wav
BAC009S0764W0123 /nfs2/speech/data/asr/aishell/data_aishell/wav/test/S0764/BAC009S0764W0123.wav
BAC009S0764W0124 /nfs2/speech/data/asr/aishell/data_aishell/wav/test/S0764/BAC009S0764W0124.wav
BAC009S0764W0125 /nfs2/speech/data/asr/aishell/data_aishell/wav/test/S0764/BAC009S0764W0125.wav
```

`text` 文件用于记录对应UTTID这条音频的标签，例如ASR标签、AST标签。FAQ标签，等等。我们在ASR的训练中对文本做了正则化，但在AST的训练中保留文本原始的形式。

**`text` 样例：**

```plaintext
BAC009S0764W0121 甚至出现交易几乎停滞的情况
BAC009S0764W0122 一二线城市虽然也处于调整中
BAC009S0764W0123 但因为聚集了过多公共资源
BAC009S0764W0124 为了规避三四线城市明显过剩的市场风险
BAC009S0764W0125 标杆房企必然调整市场战略
```

### 打包流程
输入上述wav.scp和text的路径，我们会对原始音频打包为用于训练和测试的shard文件。如果您用于对测试集打包，我们建议一个单独的测试集只打包为一个shard文件。我们默认100000条音频会写入一个shard，如果需要可以修改 `src/tools/data_package.sh`中的`num_utts_per_shard`参数。

**打包训练集：**

由于训练集通常会很大，用户可以使用 **多进程的方式进行打包，来实现加速**。通过修改 `num_threads` 来实现多进程打包。`text_norm`会对空格和英文进行normalization。
```shell
bash src/tools/data_package.sh \
  --wav_scp /nfs1/zhenlin.liang/data/training/wav.scp \
  --text /nfs1/zhenlin.liang/data/training/text \
  --write_dir /jfs/zhenlin.liang/tmp/pkg_training \
  --write_prefix data.list \
  --shuffle false \
  --text_norm true \
  --data_type shard \
  --num_threads 32
```
则会生成文件 /jfs/zhenlin.liang/tmp/pkg_training/data.list，其中内容为：
```
/jfs/zhenlin.liang/tmp/aishell_pkg_test/shards_000000000.tar
/jfs/zhenlin.liang/tmp/aishell_pkg_test/shards_000000001.tar
/jfs/zhenlin.liang/tmp/aishell_pkg_test/shards_000000002.tar
/jfs/zhenlin.liang/tmp/aishell_pkg_test/shards_000000003.tar
...
```



**打包测试集：**
```shell
bash src/tools/data_package.sh \
  --wav_scp /nfs1/zhenlin.liang/data/testsets_wavscp/test_aishell/wav.scp \
  --text /nfs1/zhenlin.liang/data/testsets_wavscp/test_aishell/text \
  --write_dir /jfs/zhenlin.liang/tmp/aishell_pkg_test \
  --write_prefix data.list \
  --shuffle false \
  --text_norm true \
  --data_type shard \
  --num_threads 1
```
则会生成文件 /jfs/zhenlin.liang/tmp/aishell_pkg_test/data.list，其中内容为：
```
/jfs/zhenlin.liang/tmp/aishell_pkg_test/shards_000000000.tar
```


## 模型训练
数据处理完成后，可以愉快的训练你的音频理解大模型了！（我们仅尝试了ASR、AST、S2S等，但实验证明可能大部分音频理解任务都能很好的完成~）

### 配置文件
核心关注两个配置文件：
- src/mooer/configs/asr_config_training.py
- src/mooer/configs/deepspeed_config_zero2.json

`asr_config_training.py` 是用于训练的配置文件 (`training config`)。当然，你也可以在任意位置创建一个新的配置文件，例如 `config.py`。在配置文件中，有一些关键参数是必须配置的：

- **self.llm_path**: `加载的LLM 大模型的路径，例如 pretrained_models/Qwen2-7B-Instruct`
- **self.encoder_path**: `加载的encoder的路径，例如 pretrained_models/paraformer_encoder/paraformer-encoder.pth，可以从我们的huggingface或modelscope下载对应模型`
- **self.output_dir**：`保存模型的路径`
- **self.train_data_path**：`训练数据的路径，例如上述打包好的 /jfs/zhenlin.liang/tmp/pkg_training/data.list`

另外，还有一些关键参数，例如：

- **self.adapter_path**：`如果提供，会加载预训练的adapter，例如可以基于我们的8万小时模型进行微调`
- **self.lora_dir**：`如果提供，会加载预训练的Lora权重，例如可以基于我们的8万小时模型进行微调`
- **self.save_merge_rank**：`默认设置为True。会将Deepspeed在不同卡保存的权重合并为一个pt文件，同时将Lora weight单独保存。`
- **self.gradient_checkpoint**：`如果打开，会对encoder进行梯度重算，可以明显降低whisper的显存占用`
- **self.find_unused_parameters**：`encoder参与训练的时候需要设置为True`
- **self.deepspeed_config**：`Deepspeed的配置文件，默认为src/mooer/configs/deepspeed_config_zero2.json`


我们的大规模数据的训练（8万小时规模）基于Deepspeed进行训练，我们在`deepspeed_config_zero2.json` 给出了我们的训练配置。当然，你也可以在任意位置创建一个新的配置文件，例如 `ds_config.py`。我们也给出了基于Zero3的配置文件：`deepspeed_config_zero3.json`。你可以在配置文件中修改训练策略、batchsize和一些显存管理机制等。

### 训练流程
修改 `train.sh` 中相关配置，进行训练。你可以通过hostfile来实现多机多卡的训练。修改 `train.sh`中的 `training_config` 为你的训练的配置文件路径。启动训练：


```shell
nohup bash train.sh > your_awesome_training_log 2>&1 &
```

你就能看到对应的训练日志如下：

```plaintext
[2024-08-15 21:08:16][root][INFO] - Training Epoch: 1/10, step 1000 lr 8.996566681120063e-05 completed (loss: 4.442868709564209, acc: 0.2647058963775635)
[2024-08-15 21:10:21][root][INFO] - Training Epoch: 1/10, step 1100 lr 9.134542298314146e-05 completed (loss: 4.4935736656188965, acc: 0.30645161867141724)
[2024-08-15 21:12:21,408] [INFO] [logging.py:96:log_dist] [Rank 0] step=600, skipped=0, lr=[9.26050416794548e-05], mom=[(0.9, 0.999)]
[2024-08-15 21:12:21,418] [INFO] [timer.py:260:stop] epoch=0/micro_step=1200/global_step=600, RunningAvgSamplesPerSec=51.18608989565034, CurrSamplesPerSec=56.79136995208808, MemAllocated=15.78GB, MaxMemAllocated=43.28GB
```

## 解码与推理

### 解码方法
模型训练完成后，我们可以得到以下ckpt。例如在你的output路径下，存在以下文件：


- asr_epoch_1_step_320001_merged
  - adapter_project.pt
  - new_llm
    - adapter_config.json
    - adapter_model.bin
    - README.md
   

核心关注一个配置文件：
- src/mooer/configs/asr_config_inference.py

`asr_config_inference.py` 是用于测试的配置文件 (`inference config`)。当然，你也可以在任意位置创建一个新的配置文件，例如 `config.py`。在配置文件中，有一些关键参数是必须配置的：

- **self.llm_path**: `加载的LLM 大模型的路径，例如 pretrained_models/Qwen2-7B-Instruct`
- **self.encoder_path**: `加载的encoder的路径，例如 pretrained_models/paraformer_encoder/paraformer-encoder.pth，可以从我们的huggingface或modelscope下载对应模型`
- **self.adapter_path**: `训练得到的adapter模型路径，例如 asr_epoch_1_step_320001_merged/adapter_project.pt`
- **self.lora_dir**: `训练得到的Lora 权重的路径，例如 asr_epoch_1_step_320001_merged/new_llm`
- **val_batch_size**: `batch解码的参数`


测试集的打包方法参考数据打包部分。一个完整的测试集包括：

- aishell
  - data.list
  - text
 

修改 `inference.sh` 来批量的解码你的测试集和计算指标。

- **test_data_dir**: `测试集的目录。例如测试集为/testsets/aishell，则这里为/testsets`
- **test_sets**: `测试集的名字，支持多个测试集，以 / 分割。例如test-clean/test-other`
- **decode_path**: `解码保存的路径`


启动解码：


```shell
nohup bash inference.sh > your_awesome_decode_log 2>&1 &
```

解码完成后，结果如下：

- your_decode_path
  - testset1
     - text
     - wer
  - testset2
     - text
     - wer
       

### 推理示例
你也可以使用wav文件来直接进行推理，参考 https://github.com/MooreThreads/MooER#-inference



