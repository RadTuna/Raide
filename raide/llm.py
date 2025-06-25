# Internal import
from speaker_profile import profile_store

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


@dataclass
class LanguageModelConfig:
    model_path: str = ""
    speaker_profile: str = "kr_firefly"
    context_window: int = 4096
    output_max_tokens: int = 512
    temperature: float = 1.0
    repeat_penalty: float = 1.0
    top_k: int = 64
    top_p: float = 0.95
    min_p: float = 0.01

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
    def __init__(self, config: LanguageModelConfig):
        self.model = ChatLlamaCpp(
            model_path=config.model_path,
            n_gpu_layers=-1,
            n_batch=512,
            n_ctx=config.context_window,
            max_tokens=config.output_max_tokens,
            temperature=config.temperature,
            repeat_penalty=config.repeat_penalty,
            top_k=config.top_k,
            top_p=config.top_p,
            streaming=False,
            f16_kv=True,
            verbose=True,
            model_kwargs={
                "chat_format": "gemma",
                "min_p": config.min_p
            }
        )

        if profile_store.is_profile_exists(config.speaker_profile):
            profile = profile_store.get_profile(config.speaker_profile)
            system_prompt = [profile.system_prompt, profile.persona]
        else:
            system_prompt = [" "]

        self.compiled_graph: CompiledGraph = self._build_graph()
        self.chat_config = {"configurable": {"thread_id": "unique_chat"}}
        self.prompt_template = ChatPromptTemplate.from_messages([
            HumanMessage(content="\n".join(system_prompt)),
            MessagesPlaceholder(variable_name="messages")
        ])
    
    async def chat(self, user_message: str) -> AsyncIterator[str]:
        if self.compiled_graph is None:
            return
        
        user_input = [HumanMessage(content=user_message)]

        async for chunk, _ in self.compiled_graph.astream(
                input={"messages": user_input},
                config=self.chat_config,
                stream_mode="messages"
            ):
            yield chunk.content

    def warmup(self):
        self.model.invoke("short response")

    async def _call_model(self, state: ModelState, config: LanguageModelConfig):
        prompt = self.prompt_template.invoke(state)
        response = await self.model.ainvoke(prompt, config)
        print(f"Response: {response}")
        return {"messages": [response]}

    def _build_graph(self) -> CompiledGraph:
        graph = StateGraph(state_schema=ModelState)
        graph.add_node("model", self._call_model)
        graph.add_edge(START, "model")

        memory = MemorySaver()
        return graph.compile(checkpointer=memory)

