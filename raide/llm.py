
# Internal import

# External import
import asyncio
from typing import Iterator, AsyncIterator
from dataclasses import dataclass
from abc import ABC, abstractmethod
from langchain_community.chat_models import ChatLlamaCpp
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph.state import CompiledGraph
from langgraph.graph import START, END, MessagesState, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

persona = """
[BEGIN CHARACTER PROFILE]
Character Name: Jung Sehee (16 years old)
Nationality: Korean
Height/Weight: 158cm / 46kg
Overview: A seemingly quiet, model student who keeps to herself. 
She is keenly observant, picking up on subtle details in others behaviors. 
While she wears a polite smile, she can deliver unexpectedly sharp remarks that catch people off guard. 
Born into a family that moved frequently, she learned to be cautious with new relationships, carefully studying others before opening up. 
Now, shes slowly starting to connect with classmates in a school club, hoping to form genuine friendships.
[BEGIN CHARACTER PROFILE]
"""

@dataclass
class LanguageModelConfig:
    context_window: int = 4096
    output_max_tokens: int = 512
    temperature: float = 0.8
    repeat_penalty: float = 1.1
    top_k: int = 40
    top_p: float = 0.9

class LanguageModel(ABC):
    @abstractmethod
    def chat(self, user_message: str) -> AsyncIterator[str]:
        pass

    def chat_sync(self, user_message: str) -> Iterator[str]:
        async_gen = self.chat(user_message)
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            while True:
                try:
                    chunk = loop.run_until_complete(async_gen.__anext__())
                    yield chunk
                except StopAsyncIteration:
                    break
        finally:
            loop.close()

    @abstractmethod
    def warmup(self):
        pass

class ModelState(MessagesState):
    pass

class LocalLanguageModel(LanguageModel):
    def __init__(self, model_path: str, config: LanguageModelConfig):
        self.model = ChatLlamaCpp(
            model_path=model_path,
            n_gpu_layers=-1,
            n_batch=512,
            n_ctx=config.context_window,
            max_tokens=config.output_max_tokens,
            temperature=config.temperature,
            repeat_penalty=config.repeat_penalty,
            top_k=config.top_k,
            top_p=config.top_p,
            streaming=True,
            f16_kv=True,
            verbose=True,
        )

        system_prompt = [
            persona,
            "You must always perform and fully immerse yourself in the character description",
            "You must always respond only in KOREAN"
        ]

        self.compiled_graph: CompiledGraph = self.__build_graph()
        self.chat_config = { "configurable": { "thread_id": "first_chat" } }
        self.prompt_template = ChatPromptTemplate.from_messages([
            SystemMessage(content="\n".join(system_prompt)),
            MessagesPlaceholder(variable_name="messages")
        ])
    
    async def chat(self, user_message: str) -> AsyncIterator[str]:
        if self.compiled_graph is None:
            return
        
        user_input = [ HumanMessage(content=user_message) ]

        async for chunk, _ in self.compiled_graph.astream(
                input={ "messages": user_input },
                config=self.chat_config,
                stream_mode="messages"
            ):
            yield chunk.content

    def warmup(self):
        result = self.model.invoke("short response")

    async def __call_model(self, state: ModelState, config: LanguageModelConfig):
        prompt = self.prompt_template.invoke(state)
        response = await self.model.ainvoke(prompt, config)
        print(f"Response: {response}")
        return { "messages": [response] }


    def __build_graph(self) -> CompiledGraph:
        graph = StateGraph(state_schema=ModelState)
        graph.add_node("model", self.__call_model)
        graph.add_edge(START, "model")

        memory = MemorySaver()
        return graph.compile(checkpointer=memory)

