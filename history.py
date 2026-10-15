import json
import redis


class History:
    def __init__(self, url, max_turns=8, ttl=3600):
        self.r = redis.Redis.from_url(url, decode_responses=True)
        self.max_turns = max_turns
        self.ttl = ttl


    def add(self, session_id, question, answer):
        key = f"chat:history:{session_id}"
        pipe = self.r.pipeline()
        pipe.rpush(key, json.dumps({"q": question, "a": answer}))
        pipe.ltrim(key, -self.max_turns, -1)
        pipe.expire(key, self.ttl)
        pipe.execute()


    def get(self, session_id):
        key = f"chat:history:{session_id}"

        messages = []
        for raw in self.r.lrange(key, 0, -1):
            t = json.loads(raw)
            messages.append({"role": "user", "content": t["q"]})
            messages.append({"role": "assistant", "content": t["a"]})
        return messages