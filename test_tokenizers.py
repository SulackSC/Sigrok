from iqbot import genai

print(type(genai.client))
print(genai.client.count_tokens("This is a test"))
