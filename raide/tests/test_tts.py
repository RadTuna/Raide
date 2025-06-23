

def test_openaudio_tts():
    from text_to_speech import OpenAudioTextToSpeech, TextToSpeechConfig

    config = TextToSpeechConfig()
    tts = OpenAudioTextToSpeech(config=config)

    ref_text = "でも、未来に起きることは依然としてはっきりしている。本当の試練はこの後にやってくるわ。若い雲騎軍の剣士は、手ごわい敵よ。"
    #sample_text = "안녕하세요. 저는 OpenAudio의 TTS입니다."
    sample_text = "だから人の手で「星の神」を殺すとは言ってない、でしょう？"

    print("TTS 시작")
    tts.create_speaker_profile(ref_voice_path="./assets/Kafka_JP.wav", ref_voice_text=ref_text)
    tts.text_to_speech(text=sample_text)
    print("TTS 완료")


test_openaudio_tts()
