import os
import itertools
from typing import List, Tuple
import groq
from groq import Groq

class GroqKeyPool:
    def __init__(self, api_keys: List[str]):
        if not api_keys:
            raise ValueError("At least one Groq API key must be provided.")
        
        # Instantiate a client per API key
        self.clients = [Groq(api_key=key.strip()) for key in api_keys if key.strip()]
        self._pool_cycle = itertools.cycle(self.clients)

    def chat_completion_with_fallback(
        self, 
        messages: List[dict], 
        models: List[str] = ["openai/gpt-oss-20b"],
        temperature: float = 0.2
    ) -> Tuple[any, str]:
        """
        Rotates across available Groq accounts and models.
        Catches 429 Rate Limits and seamlessly fails over to the next key.
        """
        last_exception = None
        
        # Try every key in the pool against each candidate model
        for model in models:
            # Iterate through all available API key clients for the current model
            for _ in range(len(self.clients)):
                client = next(self._pool_cycle)
                try:
                    response = client.chat.completions.create(
                        model=model,
                        messages=messages,
                        temperature=temperature
                    )
                    return response, model
                except groq.RateLimitError as e:
                    print(f"⚠️ Key hit rate limit on model {model}. Rotating to next account key...")
                    last_exception = e
                    continue
                except groq.APIStatusError as e:
                    # If 404 Model Not Found, break to next model in list
                    if e.status_code == 404:
                        print(f"⚠️ Model {model} not found on this endpoint. Trying next model...")
                        break
                    print(f"⚠️ Groq API Status Error: {e}")
                    last_exception = e
                    continue
                except Exception as e:
                    print(f"⚠️ Groq Exception: {e}")
                    last_exception = e
                    continue

        # If all keys and models fail
        raise last_exception or Exception("All Groq accounts and model fallbacks exhausted.")