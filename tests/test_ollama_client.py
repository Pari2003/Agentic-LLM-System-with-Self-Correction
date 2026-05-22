import asyncio
import sys
from src.models.llm_client import OllamaClient

async def main():
    print("Initializing OllamaClient...")
    client = OllamaClient()
    
    print("Checking Ollama server health...")
    health = await client.health_check()
    print("Health report:")
    print(f"  Status: {health.get('status')}")
    print(f"  Ollama URL: {health.get('ollama_url')}")
    print(f"  Available Models: {health.get('available_models')}")
    print(f"  Text model ({client.text_model}) ready: {health.get('text_model', {}).get('ready')}")
    print(f"  Embed model ({client.embed_model}) ready: {health.get('embed_model', {}).get('ready')}")
    print(f"  Vision model ({client.vision_model}) ready: {health.get('vision_model', {}).get('ready')}")
    
    if health.get('status') == 'healthy':
        print("\nAll systems green! Health check verified.")
    else:
        print("\nOllama is not running or not healthy. Please make sure Ollama is started.")
        sys.exit(1)
        
    await client.close()

if __name__ == "__main__":
    asyncio.run(main())
