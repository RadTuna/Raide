
from core.text_to_speech import OpenAITextToSpeech

tts = OpenAITextToSpeech()

sample_text = "안녕하세요. 제 이름은 OpenAI Whisper 입니다."

print("TTS 시작")
tts.text_to_speech(sample_text)
print("TTS 완료")
