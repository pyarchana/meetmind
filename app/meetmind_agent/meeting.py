"""
Live meeting state and the tools the agent uses to maintain it.

State lives in the ADK session, so it survives across turns and every write
lands in the event stream as a state delta the browser can render. Each tool
reassigns the whole list rather than mutating it in place, because ADK only
records a delta when a key is assigned.
"""

from datetime import datetime, timezone

from google.adk.tools import ToolContext

DECISIONS = "decisions"
QUESTIONS = "questions"
ACTIONS = "actions"


def _append(tool_context: ToolContext, key: str, entry: dict) -> dict:
    items = list(tool_context.state.get(key, []))
    entry["id"] = len(items) + 1
    entry["at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    items.append(entry)
    tool_context.state[key] = items
    return entry


def record_decision(decision: str, tool_context: ToolContext) -> dict:
    """Record something the meeting has agreed on.

    Call this the moment the group settles a question, so it can be recalled
    later and appears in the meeting summary.

    Args:
        decision: What was agreed, in one sentence.
    """
    return {"status": "recorded", "decision": _append(
        tool_context, DECISIONS, {"text": decision}
    )}


def record_action_item(task: str, owner: str, tool_context: ToolContext) -> dict:
    """Record a task somebody committed to during the meeting.

    Args:
        task: What needs to be done.
        owner: Who owns it. Pass "unassigned" if nobody was named.
    """
    return {"status": "recorded", "action": _append(
        tool_context, ACTIONS, {"task": task, "owner": owner, "done": False}
    )}


def record_open_question(question: str, tool_context: ToolContext) -> dict:
    """Record a question the meeting raised but did not resolve.

    Args:
        question: The unresolved question, in one sentence.
    """
    return {"status": "recorded", "question": _append(
        tool_context, QUESTIONS, {"text": question, "answered": False, "answer": None}
    )}


def resolve_open_question(question_id: int, answer: str, tool_context: ToolContext) -> dict:
    """Mark an open question as answered.

    Args:
        question_id: The id returned when the question was recorded.
        answer: How it was resolved, in one sentence.
    """
    questions = [dict(q) for q in tool_context.state.get(QUESTIONS, [])]
    for question in questions:
        if question["id"] == question_id:
            question["answered"] = True
            question["answer"] = answer
            tool_context.state[QUESTIONS] = questions
            return {"status": "resolved", "question": question}

    return {"status": "not_found", "question_id": question_id}


def get_meeting_state(tool_context: ToolContext) -> dict:
    """Return everything recorded in this meeting so far.

    Use this to answer questions about what was decided, what is still open,
    or who owns what. Answer from this, never from memory.
    """
    return {
        DECISIONS: tool_context.state.get(DECISIONS, []),
        QUESTIONS: tool_context.state.get(QUESTIONS, []),
        ACTIONS: tool_context.state.get(ACTIONS, []),
    }


TOOLS = [
    record_decision,
    record_action_item,
    record_open_question,
    resolve_open_question,
    get_meeting_state,
]
