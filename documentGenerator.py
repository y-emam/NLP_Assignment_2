from openai import OpenAI
from dotenv import load_dotenv

import os

load_dotenv()


def generate_documents(topic):

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": "You will be given a topic, generate a small paragraph about this topic.",
            },
            {
                "role": "user",
                "content": topic,
            },
        ],
    )

    print(response.choices[0].message.content)

    return response.choices[0].message.content
