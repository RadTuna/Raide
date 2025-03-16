
# Internal imports
from audio_inputer import AudioInputer, AudioData
from speech_recognition import OpenAIWhiserASR
from text_to_speech import OpenAITextToSpeech

# External imports
from typing import Annotated
from typing_extensions import TypedDict

from langchain_openai import ChatOpenAI

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages


class State(TypedDict):
    messages: Annotated[list, add_messages]


openai_model = ChatOpenAI(model = "gpt-4o", temperature = 0.7)
def process_llm(state: State):
    response = openai_model.invoke(state["messages"])
    print(f"LLM: {response}")
    return { "messages": [ openai_model.invoke(state["messages"]) ] }


graph_builder = StateGraph(State)
graph_builder.add_node("llm", process_llm)

graph_builder.add_edge(START, "llm")
graph_builder.add_edge("llm", END)

# build graph
graph = graph_builder.compile()
print(graph.get_graph().draw_ascii())

audio_inputer = AudioInputer()
asr = OpenAIWhiserASR()
tts = OpenAITextToSpeech()
while True:
    audio_data = audio_inputer.get_audio_from_mic()

    recognized_text = asr.recognize(audio_data=audio_data)
    print(f"User: {recognized_text}")
    openai_input = { "role": "user", "content": recognized_text }

    for event in graph.stream(input = {"messages": [openai_input]}):
        for value in event.values():
            print("Assistant:", value["messages"][-1].content)
            tts.text_to_speech(value["messages"][-1].content)
