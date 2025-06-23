
def test_openai_tts():
    from text_to_speech import OpenAITextToSpeech

    tts = OpenAITextToSpeech()

    sample_text = "안녕하세요. 저는 OpenAI의 위스퍼입니다."

    print("TTS 시작")
    tts.text_to_speech(sample_text)
    print("TTS 완료")

def test_openaudio_tts():
    from text_to_speech import OpenAudioTextToSpeech

    tts = OpenAudioTextToSpeech()

    ref_text = "でも、未来に起きることは依然としてはっきりしている。本当の試練はこの後にやってくるわ。若い雲騎軍の剣士は、手ごわい敵よ。"
    sample_text = "안녕하세요. 저는 OpenAudio의 TTS입니다."

    print("TTS 시작")
    tts.create_speaker_profile(ref_voice_path="./assets/Kafka_JP.wav", ref_voice_text=ref_text)
    tts.text_to_speech(text=sample_text)
    print("TTS 완료")


#test_openai_tts()
test_openaudio_tts()
