import logging
import os
from abc import ABC, abstractmethod
from typing import Optional, Any

import asyncio
import torch
from dotenv import load_dotenv
from openai import OpenAI, AsyncOpenAI
from transformers import Pipeline, pipeline


class BaseLLMClient(ABC):
    @abstractmethod
    def complete(self, prompt: str) -> tuple[Optional[str], Optional[dict]]:
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
    def __init__(self, model_id: str = "o3-mini", concurrency: int = 30) -> None:
        """
        Initialize the OpenAI LLMClient with an API key from environment variables.
        """
        load_dotenv()
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("API key not found. Please set the OPENAI_API_KEY environment variable.")

        self.client: OpenAI = OpenAI(api_key=api_key)
        self.async_client = AsyncOpenAI(api_key=api_key)
        logging.info(f"Loaded OpenAI model: {model_id}")

        self.model_id = model_id
        self.semaphore = asyncio.Semaphore(concurrency)


    def complete(self, prompt: str) -> tuple[Optional[str], Optional[dict]]:
        messages = [{"role": "user", "content": prompt}]
        response = self.client.chat.completions.create(
            model=self.model_id,
            messages=messages
        )

        usage = response.usage
        text = response.choices[0].message.content
        return text, usage

    def complete_batch(self, prompts: list[str]) -> list[str]:
        return asyncio.run(self._complete_batch_async(prompts))

    async def _complete_batch_async(self, prompts: list[str]) -> list[tuple[Optional[str], Optional[dict]]]:
        async def _call(p: str) -> tuple[Optional[str], Optional[dict]]:
            async with self.semaphore:
                resp = await self.async_client.chat.completions.create(
                    model=self.model_id,
                    messages=[{"role": "user", "content": p}]
                )
            text = resp.choices[0].message.content
            usage = resp.usage
            return text, usage

        tasks = [asyncio.create_task(_call(p)) for p in prompts]
        return await asyncio.gather(*tasks)


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
        logging.info(f"HuggingFaceClient initialized with max_new_tokens={max_new_tokens}, batch_size={batch_size}, model_id={model_id}")

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
        logging.debug("Sending prompt to Hugging Face model: %s", messages)

        outputs = self.pipeline(messages,
                                max_new_tokens=self.max_new_tokens,
                                temperature=0.6)
        logging.debug("Raw outputs from sequential pipeline: %s", outputs)

        try:
            return outputs[0]['generated_text'][-1]["content"]
        except Exception as e:
            raise RuntimeError(f"Error extracting generated text: {e}")

    def complete_batch(self, prompts: list[str]) -> list[str]:
        messages = [[{"role": "user", "content": prompt}] for prompt in prompts]
        logging.debug("Sending batch of prompts to Hugging Face model: %s", messages)

        outputs = self.pipeline(messages,
                                max_new_tokens=self.max_new_tokens,
                                batch_size=self.batch_size,
                                temperature=0.6)
        logging.debug("Raw outputs from batch pipeline: %s", outputs)

        try:
            return [output[0]['generated_text'][-1]["content"] for output in outputs]
        except Exception as e:
            raise RuntimeError(f"Error extracting generated text in batch: {e}")


class DeepSeekClient(BaseLLMClient):
    """
    Async DeepSeek client using AsyncOpenAI under the hood but exposes
    the sync interface for compatibility with a pipeline.
    """
    def __init__(self, concurrency: int = 30, model_id: str = "deepseek-reasoner"):
        load_dotenv()
        api_key = os.getenv('DEEPSEEK_API_KEY')
        if not api_key:
            raise ValueError("DEEPSEEK_API_KEY not found in environment variables.")
        self.async_client = AsyncOpenAI(api_key=api_key, base_url="https://api.deepseek.com")
        self.model_id = model_id
        self.semaphore = asyncio.Semaphore(concurrency)

    def complete(self, prompt: str) -> tuple[str, Any]:
        """
        Single-prompt call: wraps the async call in asyncio.run
        Returns (response_text, usage)
        """
        return asyncio.run(self._complete_async(prompt))

    async def _complete_async(self, prompt: str) -> tuple[str, Any]:
        async with self.semaphore:
            resp = await self.async_client.chat.completions.create(
                model=self.model_id,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                stream=False,
            )
        usage = resp.usage
        text = resp.choices[0].message.content
        return text, usage

    def complete_batch(self, prompts: list[str]) -> list[tuple[str, Any]]:
        """
        Batch call: runs all prompts concurrently within a single event loop
        Returns a list of (response_text, usage) tuples
        """
        return asyncio.run(self._complete_batch_async(prompts))

    async def _complete_batch_async(self, prompts: list[str]) -> list[tuple[str, Any]]:
        async def _call(p: str) -> tuple[str, Any]:
            async with self.semaphore:
                resp = await self.async_client.chat.completions.create(
                    model=self.model_id,
                    messages=[{"role": "user", "content": p}],
                    temperature=0.1,
                    stream=False,
                )
            text = resp.choices[0].message.content
            usage = resp.usage
            return text, usage

        tasks = [asyncio.create_task(_call(p)) for p in prompts]
        return await asyncio.gather(*tasks)
