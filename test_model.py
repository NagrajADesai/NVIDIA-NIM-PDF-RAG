from openai import OpenAI
from dotenv import load_dotenv
import os


load_dotenv(dotenv_path=".env")

NVIDIA_API_KEY = os.getenv('NVIDIA_API_KEY')
os.environ['NVIDIA_API_KEY'] = NVIDIA_API_KEY


MODEL = "openai/gpt-oss-20b"
client = OpenAI(
  base_url = "https://integrate.api.nvidia.com/v1",
  api_key = NVIDIA_API_KEY
)

completion = client.chat.completions.create(
  model=MODEL,
  messages=[{"role":"user","content":""}],
  temperature=1,
  top_p=1,
  max_tokens=1024,
  stream=True
)

for chunk in completion:
  reasoning = getattr(chunk.choices[0].delta, "reasoning_content", None)
  if reasoning:
    print(reasoning, end="")
  if chunk.choices[0].delta.content is not None:
    print(chunk.choices[0].delta.content, end="")

