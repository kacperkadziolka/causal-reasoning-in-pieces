import logging
import os
from abc import ABC, abstractmethod

import torch
from dotenv import load_dotenv
from openai import OpenAI
from transformers import Pipeline, pipeline


class BaseLLMClient(ABC):
    @abstractmethod
    def complete(self, prompt: str) -> str:
        """
        Sends the prompt to the LLM and returns the response as a string.
        """
        pass

    @abstractmethod
    def complete_batch(self, prompts: list[str]) -> list[str]:
        """
        Sends a list of prompts to the LLM and returns a list of responses.
        Clients that support true batch execution should override this, otherwise throw an exception.
        """
        pass


class OpenAIClient(BaseLLMClient):
    def __init__(self, model_id: str = "o3-mini") -> None:
        """
        Initialize the OpenAI LLMClient with an API key from environment variables.
        """
        load_dotenv()
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("API key not found. Please set the OPENAI_API_KEY environment variable.")

        self.client: OpenAI = OpenAI(api_key=api_key)
        logging.info(f"Loaded OpenAI model: {model_id}")

        self.model_id = model_id


    def complete(self, prompt: str) -> str:
        messages = [{"role": "user", "content": prompt}]
        response = self.client.chat.completions.create(
            model=self.model_id,
            messages=messages
        )
        return response.choices[0].message.content

    def complete_batch(self, prompts: list[str]) -> list[str]:
        raise NotImplementedError("Batched completion is not supported for OpenAIClient.")


class HuggingFaceClient(BaseLLMClient):
    def __init__(self, max_new_tokens: int, batch_size: int, model_id: str = "deepseek-ai/DeepSeek-R1-Distill-Llama-70B") -> None:
        """
        Load the Hugging Face model during initialization.
        Make sure that the user is authenticated into huggingface hub.

        :param max_new_tokens: The maximum number of new tokens to generate.
        :param batch_size: The batch size for processing, suggested maximum of 4, depending on the gpu.
        :param model_id: The model ID to load from Hugging Face hub.
        """
        self.max_new_tokens = max_new_tokens
        self.batch_size = batch_size

        # Load the model from Hugging Face hub
        try:
            self.pipeline: Pipeline = pipeline(
                "text-generation",
                model=model_id,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )
            logging.info(f"Loaded Hugging Face model: {model_id}")
        except Exception as e:
            raise RuntimeError(f"Failed to load {model_id} model from huggingface hub: {e}."
                               f"Make sure you are authenticated in to the huggingface hub.")

    def complete(self, prompt: str) -> str:
        messages = [{"role": "user", "content": prompt}]
        outputs = self.pipeline(messages, max_new_tokens=self.max_new_tokens)
        return outputs[0]['generated_text'][-1]["content"]

    def complete_batch(self, prompts: list[str]) -> list[str]:
        messages = [{"role": "user", "content": prompt} for prompt in prompts]
        outputs = self.pipeline(messages, max_new_tokens=self.max_new_tokens, batch_size=self.batch_size)
        return [output['generated_text'][-1]["content"] for output in outputs]
