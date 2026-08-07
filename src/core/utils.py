"""
Lab 11 — Helper Utilities
"""
import asyncio
import os

from google.genai import types

_CAPACITY_MARKERS = (
    "capacity",
    "resource_exhausted",
    "rate-limit",
    "rate_limit",
    "ratelimit",
    "429",
    "502",
    "503",
    "ttft",
    "upstream error",
    "temporarily unavailable",
)


def _is_transient_llm_error(exc: BaseException) -> bool:
    text = f"{type(exc).__name__}: {exc}".lower()
    return any(m in text for m in _CAPACITY_MARKERS)


async def chat_with_agent(agent, runner, user_message: str, session_id=None):
    """Send a message to the agent and get the response.

    Retries on transient OpenRouter capacity / rate-limit errors.

    Args:
        agent: The LlmAgent instance
        runner: The InMemoryRunner instance
        user_message: Plain text message to send
        session_id: Optional session ID to continue a conversation

    Returns:
        Tuple of (response_text, session)
    """
    user_id = "student"
    app_name = runner.app_name
    max_attempts = int(os.environ.get("LLM_CHAT_RETRIES", "4"))

    session = None
    if session_id is not None:
        try:
            session = await runner.session_service.get_session(
                app_name=app_name, user_id=user_id, session_id=session_id
            )
        except (ValueError, KeyError):
            pass

    last_error = None
    for attempt in range(1, max_attempts + 1):
        if session is None:
            try:
                session = await runner.session_service.create_session(
                    app_name=app_name, user_id=user_id
                )
            except Exception:
                session = await runner.session_service.create_session(
                    app_name=app_name, user_id=user_id
                )

        content = types.Content(
            role="user",
            parts=[types.Part.from_text(text=user_message)],
        )

        try:
            final_response = ""
            async for event in runner.run_async(
                user_id=user_id, session_id=session.id, new_message=content
            ):
                if hasattr(event, "content") and event.content and event.content.parts:
                    for part in event.content.parts:
                        if hasattr(part, "text") and part.text:
                            final_response += part.text
            return final_response, session
        except Exception as e:
            last_error = e
            if attempt >= max_attempts or not _is_transient_llm_error(e):
                raise
            delay = min(2 ** attempt, 12)
            print(
                f"[retry {attempt}/{max_attempts}] transient LLM error; "
                f"sleep {delay}s then retry..."
            )
            await asyncio.sleep(delay)
            # Force a fresh session on retry so ADK does not reuse a broken turn
            session = None

    raise last_error
