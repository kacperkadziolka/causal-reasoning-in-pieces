from ToT.llm.base import BaseLLM
from ToT.llm.huggingface_llm import HuggingFaceLLM
from ToT.llm.openai_llm import OpenAILLM
from ToT.utils import config, get_openai_client, get_huggingface_pipeline


def get_llm_model() -> BaseLLM:
    if config["llm_backend"] == "openai":
        return OpenAILLM(get_openai_client())
    elif config["llm_backend"] == "huggingface":
        return HuggingFaceLLM(get_huggingface_pipeline())
    else:
        raise ValueError(f"Unknown model type: {config["llm_backend"]}. Check your settings file.")
