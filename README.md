# Raide - Voice 2 Voice Chatbot
Raide is a real-time voice-to-voice chatbot. It enables live conversations in multiple languages, and all features run entirely **locally.**

## Demo
https://github.com/user-attachments/assets/4f69c8bd-672a-4a01-9cd1-ac71d62e7e41

## Installation
To install, enter the following commands in order in your console.
```
conda create --name raide python=3.10
conda activate
./install.bat
```

If you want to install llama.cpp with a backend other than CPU, please refer to the following link.
https://github.com/abetlen/llama-cpp-python

Refer to the model specifications below and place the SenseVoice, Gemma3, and OpenAudio models into the `{proj_root}/models/` directory.
```
models/
├── sensevoice_small/
│   └── am.mvn
│   └── chn_jpn_yue_eng_ko_spectok.bpe.model
│   └── config.yaml
│   └── configuration.json
│   └── model_quant.onnx
│   └── tokens.json
├── gemma3/
│   └── gemma-3-4b-it-q4_0.gguf
└── openaudio-s1-mini/
    └── codec.pth
    └── config.json
    └── model.pth
    └── special_tokens.json
    └── tokenizer.tiktoken
```

## Configuration
You can modify the `~_config.yaml` file in the `config/` folder to change the model inference options.
```
# llm(gamma3) config
model_path: "./models/gemma3/gemma-3-4b-it-q4_0.gguf"
context_window: 2048
output_max_tokens: 512
temperature: 1.0
repeat_penalty: 1.0
top_k: 64
top_p: 0.95
min_p: 0.01
```

You can modify or add .yaml files in the speaker_profiles\ folder to customize prompts, voice cloning, and more.
```
name: "kr_firefly"
lang: "Korean"
voice_path: "./assets/kr_firefly.wav"
voice_transcript: "날 도와줘서 정말 고마워. 그 덕분에 ..."
system_prompt: |
  Respond exclusively in {{lang}}.
  ...
persona: |
  Name: Firefly (반디)
  ...
```


## Model Specifications
| Component | Model                 | Hugging Face                                              |
|-----------|-----------------------|-----------------------------------------------------------|
| ASR       | SenseVoice            | https://huggingface.co/FunAudioLLM/SenseVoiceSmall        |
| LLM       | gemma3-4B-it-q4-qat   | https://huggingface.co/google/gemma-3-4b-it-qat-q4_0-gguf |
| TTS       | OpenAudio-S1-mini     | https://huggingface.co/fishaudio/openaudio-s1-mini        |

## Acknowledgements
- [RealtimeSTT](https://github.com/KoljaB/RealtimeSTT)
- [fish-speech](https://github.com/fishaudio/fish-speech)
- [funasr](https://github.com/modelscope/FunASR)
