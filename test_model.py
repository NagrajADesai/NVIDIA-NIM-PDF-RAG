from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()
NVIDIA_API_KEY = os.getenv('NVIDIA_API_KEY')
os.environ['NVIDIA_API_KEY'] = NVIDIA_API_KEY


client = OpenAI(
  base_url = "https://integrate.api.nvidia.com/v1",
  api_key = NVIDIA_API_KEY
)

completion = client.chat.completions.create(
  model="meta/llama-3.1-8b-instruct",
  messages=[{"role":"user","content":"can you tell me what is the difference between you and your base model"}],
  temperature=0.2,
  top_p=0.7,
  max_tokens=512,
  stream=True
)

for chunk in completion:
  if chunk.choices[0].delta.content is not None:
    print(chunk.choices[0].delta.content, end="")