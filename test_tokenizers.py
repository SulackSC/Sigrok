# s = "This is a test"

# t1 = AutoTokenizer.from_pretrained("Xenova/gpt-4o")
# t2 = AutoTokenizer.from_pretrained("Xenova/claude-tokenizer")

# print("GPT-4o token count:", len(t1.encode(s)))
# print("Claude token count:", len(t2.encode(s)))

from iqbot import genai

print(type(genai.client))
print(genai.client.count_tokens("This is a test"))


async def main():
    system_prompt = "Respond with 'hello <name>' for the provided username'"
    prompt = "Adam"
    response = await genai.client.send_prompt(None, system_prompt, prompt)
    print(response[0:1999])


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
