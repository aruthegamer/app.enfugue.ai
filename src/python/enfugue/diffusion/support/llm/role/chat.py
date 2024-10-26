from __future__ import annotations

from typing import List, Optional

from enfugue.diffusion.support.llm.role.base import Role, MessageDict

__all__ = ["CasualChat"]

class CasualChat(Role):
    """
    This class controls the behavior for use in casual chat, where the bot may not always respond.
    """
    role_name = "casual-chat"

    @property
    def system_introduction(self) -> str:
        """
        The message told to the bot at the beginning instructing it
        """
        return "You are a helpful and friendly chatbot. A user will ask you an initial question, and you will respond. After that, more users may enter the chat and discuss your answer, or ask you more questions. You can respond to them as well. You don't have to respond to every message, especially if the user appears to be talking directly to someone else. When you do not want to respond, return an ellipsis (...)."
