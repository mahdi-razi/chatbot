from langfuse.openai import OpenAI


class Generator:
    def __init__(self, base_url, name='Qwen/Qwen3-8B-AWQ'):
        self.name = name
        self.client = OpenAI(base_url=base_url, api_key='EMPTY')

    def __call__(self, prompt, max_new_tokens=2048, schema=None, name="generate"):
        kwargs = {}
        if schema is not None:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "output", "schema": schema},
            }
        res = self.client.chat.completions.create(
            model=self.name,
            messages=prompt,
            max_tokens=max_new_tokens,
            temperature=0,
            extra_body={
                "chat_template_kwargs": {"enable_thinking": False},
            },
            name=name,
            **kwargs,
        )
        return res.choices[0].message.content