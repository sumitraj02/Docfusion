import asyncio
import os
import google.generativeai as genai

from dotenv import load_dotenv


class ModelGemini:
    def __init__(self):
        load_dotenv()
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not self.gemini_api_key:
            raise ValueError("API key for GEMINI is not set in the .env file.")
        os.environ["GEMINI_API_KEY"] = self.gemini_api_key
        genai.configure(api_key=self.gemini_api_key)

    async def gemini_response(self, prompt):
        model = genai.GenerativeModel("gemini-1.5-flash-8b")
        response = model.generate_content(prompt)
        await asyncio.sleep(1)
        return response.text

def main():
    model = ModelGemini()

    async def run():
        prompt = input("Enter your prompt: ")
        response = await model.gemini_response(prompt)
        print("Response:", response)

    asyncio.run(run())

if __name__ == "__main__":
    main()
