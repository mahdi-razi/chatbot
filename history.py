import json
import redis
from errors import HistoryError


class History:
    def __init__(self, url, max_turns=8, ttl=3600):
        self.r = redis.Redis.from_url(url, decode_responses=True, socket_timeout=2)
        self.max_turns = max_turns
        self.ttl = ttl


    def add(self, session_id, question, answer):
        key = f"chat:history:{session_id}"
        try:
            pipe = self.r.pipeline()
            pipe.rpush(key, json.dumps({"q": question, "a": answer}))
            pipe.ltrim(key, -self.max_turns, -1)
            pipe.expire(key, self.ttl)
            pipe.execute()
        except Exception as e:
            raise HistoryError(f"Failed to save history for session '{session_id}'") from e

    def get(self, session_id):
        key = f"chat:history:{session_id}"
        
        try:
            records = self.r.lrange(key, 0, -1)

        except Exception as e:
            raise HistoryError(f'Failed to retrieve history for session {session_id}') from e
        
        messages = []

        try:
            for raw in records:
                t = json.loads(raw)
                messages.append({"role": "user", "content": t["q"]})
                messages.append({"role": "assistant", "content": t["a"]})
                
        except Exception as e:
            raise HistoryError(f'Failed to parse history for session {session_id}') from e
        
        return messages