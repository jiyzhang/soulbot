import io
import re
import json
import time
import wave
import requests
import simpleaudio as sa

from mlx_lm import load, generate
from f5_tts_mlx.generate import generate as f5_generate

import mlx_whisper as whisper

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def generate_speech(text):
    # generate speech with f5-tts-mlx
    time_ckpt = time.time()
    audio = f5_generate(text)
    print("Audio Generation Time: %d ms\n" % ((time.time() - time_ckpt) * 1000))


def split_text(text):
    sentence_endings = ['！', '。', '？']
    for punctuation in sentence_endings:
        text = text.replace(punctuation, punctuation + '\n')
    pattern = r'\[.*?\]'
    text = re.sub(pattern, '', text)
    return text


"""
for voice recognition
"""

import time
import wave
import queue
import struct
import threading
import subprocess

import pyaudio

from langchain.callbacks.base import BaseCallbackHandler, BaseCallbackManager

LANG = "CN" # CN for Chinese, EN for English
DEBUG = True

# Model Configuration
#WHISP_PATH = "models/whisper-large-v3"
WHISPER_PATH = "./models/whisper-large-v3-mlx"

# Recording Configuration
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
SILENCE_THRESHOLD = 500
SILENT_CHUNKS = 2 * RATE / CHUNK  # two seconds of silence marks the end of user voice input
MIC_IDX = 0 # Set microphone id. Use tools/list_microphones.py to see a device list.

def compute_rms(data):
    # Assuming data is in 16-bit samples
    format = "<{}h".format(len(data) // 2)
    ints = struct.unpack(format, data)

    # Calculate RMS
    sum_squares = sum(i ** 2 for i in ints)
    rms = (sum_squares / len(ints)) ** 0.5
    return rms

def record_audio():
    audio = pyaudio.PyAudio()
    stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, input_device_index=MIC_IDX, frames_per_buffer=CHUNK)

    silent_chunks = 0
    audio_started = False
    frames = []

    while True:
        data = stream.read(CHUNK)
        frames.append(data)
        rms = compute_rms(data)
        if audio_started:
            if rms < SILENCE_THRESHOLD:
                silent_chunks += 1
                if silent_chunks > SILENT_CHUNKS:
                    break
            else:
                silent_chunks = 0
        elif rms >= SILENCE_THRESHOLD:
            audio_started = True

    stream.stop_stream()
    stream.close()
    audio.terminate()

    # save audio to a WAV file
    with wave.open('recordings/output.wav', 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(audio.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))

class VoiceOutputCallbackHandler(BaseCallbackHandler):
    def __init__(self):
        self.generated_text = ""
        self.lock = threading.Lock()
        self.speech_queue = queue.Queue()
        self.worker_thread = threading.Thread(target=self.process_queue)
        self.worker_thread.daemon = True
        self.worker_thread.start()
        self.tts_busy = False

    def on_llm_new_token(self, token, **kwargs):
        # Append the token to the generated text
        with self.lock:
            self.generated_text += token

        # Check if the token is the end of a sentence
        if token in ['.', '。', '!', '！', '?', '？']:
            with self.lock:
                # Put the complete sentence in the queue
                self.speech_queue.put(self.generated_text)
                self.generated_text = ""

    def process_queue(self):
        while True:
            # Wait for the next sentence
            text = self.speech_queue.get()
            if text is None:
                self.tts_busy = False
                continue
            self.tts_busy = True
            self.text_to_speech(text)
            self.speech_queue.task_done()
            if self.speech_queue.empty():
                self.tts_busy = False

    def text_to_speech(self, text):
        try:
            generate_speech(text)
        except Exception as e:
            print(f"Error in text-to-speech: {e}")


"""
end
"""

if __name__ == "__main__":
    with open('./models/SoulChat2.0-Qwen2-7B/tokenizer_config.json', 'r') as file:
        tokenizer_config = json.load(file)

    model, tokenizer = load(
        "./models/SoulChat2.0-Qwen2-7B/",
        tokenizer_config=tokenizer_config
    )

    # record_audio()

    # Create an instance of the VoiceOutputCallbackHandler
    voice_output_handler = VoiceOutputCallbackHandler()

    # Create a callback manager with the voice output handler
    callback_manager = BaseCallbackManager(handlers=[voice_output_handler])

    user_input = ""
    # sys_msg = 'You are a helpful assistant'
    with open('./text/chat_template.txt', 'r') as template_file:
        template = template_file.read()

    try:
        while True:
            if voice_output_handler.tts_busy:  # Check if TTS is busy
                continue  # Skip to the next iteration if TTS is busy 
            try:
                print("Listening...")
                record_audio()
                print("Transcribing...")
                time_ckpt = time.time()
                user_input = whisper.transcribe("recordings/output.wav", path_or_hf_repo=WHISPER_PATH)["text"]
                print("%s: %s (Time %d ms)" % ("Guest", user_input, (time.time() - time_ckpt) * 1000))
            
            except subprocess.CalledProcessError:
                print("voice recognition failed, please try again")
                continue
            
            # question = user_input
            prompt = template.replace("{usr_msg}", user_input)
            print("%s: %s" % ("问题", user_input))
            # user_input = split_text(user_input)
            # generate_speech(question, 9880)
            
            time_ckpt = time.time()
            response = generate(
                model,
                tokenizer,
                prompt=prompt,
                temp=0.3,
                max_tokens=500,
                verbose=False
            )

            print("%s: %s (Time %d ms)\n" % ("知音在线", response, (time.time() - time_ckpt) * 1000))
            response = split_text(response)
            voice_output_handler.text_to_speech(response)
            # generate_speech(response)
    except KeyboardInterrupt:
        pass
