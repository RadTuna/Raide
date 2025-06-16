
from llm import LocalLanguageModel, LanguageModelConfig

config = LanguageModelConfig()
config.temperature = 1.0
llm = LocalLanguageModel(model_path="./models/gemma3/gemma-3-4b-it-Q8_0.gguf", config=config)

test_message = "안녕하세요. 오늘 날씨가 좋네요. 당신의 소개를 해주세요."

print("start llm test")

llm.warmup()

for chunk in llm.chat_sync(test_message):
    print(chunk, end="")

print("end llm test")
