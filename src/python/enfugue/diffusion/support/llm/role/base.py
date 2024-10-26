from __future__ import annotations

from typing import List, Dict, TypedDict, Any, Optional

__all__ = ["Role", "MessageDict"]

class MessageDict(TypedDict):
    role: str
    content: str

class Role:
    """
    This class allows for defining roles for LLMs to adopt
    """
    role_name = "default"
    remove_notes = False
    single_line = False

    def __init__(self, **kwargs: Any) -> None:
        pass

    def format_input(self, message: Optional[str], **kwargs: Optional[str]) -> str:
        """
        Given user input, format the message to the bot
        """
        return "" if message is None else message

    def format_output(self, message: str, **kwargs: Optional[str]) -> str:
        """
        Given bot output, format the message to the user
        """
        if self.remove_notes:
            lines = message.splitlines()
            message = "\n".join([
                line for line in lines if not line.lower().strip("() ").startswith("note:")
            ])
        if self.single_line and message:
            message = message.splitlines()[0]
        return message

    @property
    def use_system(self) -> bool:
        """
        Whether to use the system prompts
        """
        return True

    @property
    def use_chat(self) -> bool:
        """
        Whether to use the conversation structure
        """
        return True

    @property
    def system_greeting(self) -> str:
        """
        The greeting displayed at the start of conversations
        """
        return "Hello, I am your personal assistant. How can I help today?"

    @property
    def system_introduction(self) -> str:
        """
        The message told to the bot at the beginning instructing it
        """
        return "You are a general-purpose assistant bot. A user will prompt you with various questions or instructions, and you are to respond as helpfully as possible to the best of your ability. At times, the user may request that you perform a different function. You should interpret this request and comply by operating along the user's provided guidelines."

    @property
    def introduction(self) -> str:
        """
        Gets either the set or default introduction
        """
        if hasattr(self, "_introduction"):
            return self._introduction
        return self.system_introduction

    @introduction.setter
    def introduction(self, introduction: Optional[str]) -> None:
        """
        Sets the introduction (or resets to default)
        """
        if not introduction:
            try:
                delattr(self, "_introduction")
            except AttributeError:
                pass
        else:
            self._introduction = introduction

    @property
    def system_rules(self) -> List[str]:
        """
        The rules given to the regular bot.
        """
        return []

    @property
    def system_message(self) -> str:
        """
        The formatted message told to the bot at the beginning instructing it
        """
        message = self.introduction
        if self.system_rules:
            message += "\n\nThere are a few rules to follow:\n{0}".format(
                "\n".join([f"  - {rule}" for rule in self.system_rules])
            )
        return message

    @property
    def max_new_tokens(self) -> int:
        """
        The maximum length of the response
        """
        return 2048

    @property
    def kwargs(self) -> Dict[str, Any]:
        """
        The arguments passed to the invocation
        """
        return {
            "max_new_tokens": self.max_new_tokens
        }

    @property
    def system_examples(self) -> List[MessageDict]:
        """
        Examples given to the system to refine behavior
        """
        return []

    @property
    def system_conversation(self) -> List[MessageDict]:
        """
        The conversation to begin the invocation
        """
        return [{
            "role": "system",
            "content": self.system_message
        }] + self.system_examples # type: ignore[return-value]
