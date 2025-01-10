# 心理大模型语音问答

### 使用的工具

- 语音转文字
    - mlx-whisper https://github.com/ml-explore/mlx-examples/tree/main/whisper
- 文字转语音
    - F5-TTS-MLX https://github.com/lucasnewman/f5-tts-mlx
- 心理咨询大模型
    - SoulChat2.0 心理咨询师数字孪生框架 https://github.com/scutcyr/SoulChat2.0

### 安装步骤

1. 创建 env
    
    ```bash
    conda create -n soulbot python=3.10
    ```
    
2. 安装系统依赖
    1. Install [`ffmpeg`](https://ffmpeg.org/):
        
        ```bash
        # on macOS using Homebrew (https://brew.sh/)
        brew install ffmpeg
        ```
        
    2. Install the `portaudio`:
        
        ```bash
        brew install portaudio
        ```
        
    
3. 代码: https://github.com/jiyzhang/soulbot
    
    ```bash
    git clone https://github.com/jiyzhang/soulbot
    cd soulbot
    ```

4. 安装python依赖
    
    ```bash
    pip install -r requirements.txt
    ``` 

### 模型下载

1. 下载 whisper-large-v3-mlx
    1. https://huggingface.co/mlx-community/whisper-large-v3-mlx
    2. 安装
        
        ```bash
        mkdir -p models/whisper-large-v3-mlx
        pip install huggingface-hub
        export HF_ENDPOINT=https://hf-mirror.com
        huggingface-cli download mlx-community/whisper-large-v3-mlx --local-dir ./models/whisper-large-v3-mlx
        ```
        
2. 安装 soulchat2.0
    
    使用 SoulChat2.0-Qwen2-7B, 这个模型基于 Qwen2-7b-Instruct。这个模型需要从魔搭下载
    
    ```bash
    mkdir -p models/SoulChat2.0-Qwen2-7B
    #安装ModelScope
    pip install modelscope
    modelscope download --model 'YIRONGCHEN/SoulChat2.0-Qwen2-7B' --include '*' --local_dir ./models/SoulChat2.0-Qwen2-7B/
    ```
    

### 运行

```bash
python start_soulbot.py
Listening...
Transcribing...
Guest: 大夫我最近睡眠不好经常做噩梦请问怎么办 (Time 3539 ms)
问题: 大夫我最近睡眠不好经常做噩梦请问怎么办
[Warning] Specifying sampling arguments to ``generate_step`` is deprecated. Pass in a ``sampler`` instead.
知音在线: 你好，首先感谢你信任我并分享你的困扰。睡眠不好和做噩梦肯定让你感到非常疲惫和困扰。我们可以一起探讨一些可能的解决方法。你能告诉我噩梦通常发生在什么情况下吗？ (Time 8514 ms)

Fetching 3 files: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████| 3/3 [00:00<00:00, 31223.11it/s]
Fetching 2 files: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████| 2/2 [00:00<00:00, 24745.16it/s]
Got reference audio with duration: 5.33 seconds
Generated 9.65s of audio in 0:00:09.966128.
Audio Generation Time: 21713 ms

Listening...
Transcribing...
Guest: 一般发生在比较紧张的时候 (Time 2457 ms)
问题: 一般发生在比较紧张的时候
[Warning] Specifying sampling arguments to ``generate_step`` is deprecated. Pass in a ``sampler`` instead.
知音在线: 听起来，你在紧张的时候会经历一些情绪上的波动。能分享一下，当你感到紧张时，通常是在什么样的情况下吗？ (Time 8089 ms)

Fetching 3 files: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████| 3/3 [00:00<00:00, 16958.10it/s]
Fetching 2 files: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████| 2/2 [00:00<00:00, 24174.66it/s]
Got reference audio with duration: 5.33 seconds
Generated 7.29s of audio in 0:00:08.437943.
Audio Generation Time: 17357 ms

Listening...
Transcribing...
Guest: 比如说这个作业没有写完的时候 (Time 1886 ms)
问题: 比如说这个作业没有写完的时候
[Warning] Specifying sampling arguments to ``generate_step`` is deprecated. Pass in a ``sampler`` instead.
知音在线: 你好，我在这里愿意聆听你的困扰。关于作业没有写完的情况，这让你感到怎样？ (Time 6806 ms)

Fetching 3 files: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████| 3/3 [00:00<00:00, 30916.25it/s]
Fetching 2 files: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████| 2/2 [00:00<00:00, 20020.54it/s]
Got reference audio with duration: 5.33 seconds
Generated 5.97s of audio in 0:00:06.266798.
Audio Generation Time: 13740 ms

Listening...
^C
```