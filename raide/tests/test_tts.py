
def test_openai_tts():
    from text_to_speech import OpenAITextToSpeech

    tts = OpenAITextToSpeech()

    sample_text = "안녕하세요. 저는 OpenAI의 위스퍼입니다."

    print("TTS 시작")
    tts.text_to_speech(sample_text)
    print("TTS 완료")

def test_cosyvoice_tts():
    from text_to_speech import CosyVoiceTextToSpeech

    tts = CosyVoiceTextToSpeech()

    sample_text = "안녕하세요, 저는 코지보이스입니다."

    print("TTS 시작")
    tts.text_to_speech(sample_text)
    print("TTS 완료")


#test_openai_tts()
test_cosyvoice_tts()
