# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------
#pylint: skip-file
import copy
from typing import List, Tuple

from azure.ai.generative.synthetic.simulator._conversation import (
    ConversationBot,
    ConversationTurn,
)

class HistoryIterator:
  def __init__(self, history, role):
      self.history = [m for m in history["messages"] if m["role"] == role]

  def __iter__(self):
    self.count = 1
    return self

  def __next__(self):
    if self.count <= len(self.history):
      self.count += 1
      return self.history[self.count-2]
    else:
      raise StopIteration
      

class ConversationHistoryBot(ConversationBot):
    def __init__(self, conversation_history, *args, **kwargs):
        self.history_iterator = iter(HistoryIterator(conversation_history, "user"))
        super().__init__(*args, **kwargs)

    async def generate_response(
        self,
        session: "RetryClient",  # type: ignore[name-defined]
        conversation_history: List[ConversationTurn],
        max_history: int,
        turn_number: int = 0,
    ) -> Tuple[dict, dict, int, dict]:
        next_message = next(self.history_iterator, None)
        if next_message is None:
            return await super().generate_response(
                session=session,
                conversation_history=conversation_history,
                max_history=max_history,
                turn_number=turn_number
            )

        time_taken = 0

        finish_reason = ["stop"]

        parsed_response = {"samples": [next_message["content"]], "finish_reason": finish_reason, "id": None}
        full_response = parsed_response
        return parsed_response, {}, time_taken, full_response

